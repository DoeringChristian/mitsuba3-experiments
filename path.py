import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    dr.set_flag(dr.JitFlag.KernelHistory, True)


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class Path(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        self.max_depth = props.get("max_depth", def_value=16)
        self.rr_depth = props.get("rr_depth", def_value=4)
        super().__init__(props)

    def render_sample(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock,
        aovs: list[mi.Float32],
        pos: mi.Vector2f,
        diff_scale_factor: mi.ScalarFloat32,
        active: mi.Bool = True,
    ):
        film = sensor.film()
        has_alpha = mi.has_flag(film.flags(), mi.FilmFlags.Alpha)
        box_filter = film.rfilter().is_box_filter()

        scale = 1.0 / mi.ScalarVector2f(film.crop_size())
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale

        sample_pos = pos + sampler.next_2d(active)
        adjusted_pos = dr.fma(sample_pos, scale, offset)

        apperature_sample = mi.Point2f(0.5)
        if sensor.needs_aperture_sample():
            apperature_sample = sampler.next_2d(active)

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0.0:
            time += sampler.next_1d(active) * sensor.shutter_open_time()

        wavelength_sample = 0.0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d(active)

        ray, ray_weight = sensor.sample_ray_differential(
            time, wavelength_sample, adjusted_pos, apperature_sample
        )

        if ray.has_differentials:
            ray.scale_differential(diff_scale_factor)

        medium = sensor.get_medium()

        spec, valid = self.sample(scene, sampler, ray, medium, active)

        spec_u = mi.unpolarized_spectrum(ray_weight * spec)

        if mi.has_flag(film.flags(), mi.FilmFlags.Special):
            film.prepare_sample(
                spec_u,
                ray.wavelengths,
                aovs,
                1.0,
                dr.select(valid, mi.Float32(1.0), mi.Float32(0.0)),
                valid,
            )
        else:
            rgb = mi.Color3f()
            if mi.is_spectral:
                rgb = mi.spectrum_list_to_srgb(spec_u, ray.wavelengths, active)
            elif mi.is_monochromatic:
                rgb = spec_u.x
            else:
                rgb = spec_u

            aovs[0] = rgb.x
            aovs[1] = rgb.y
            aovs[2] = rgb.z

            if has_alpha:
                aovs[3] = dr.select(valid, mi.Float32(1.0), mi.Float32(0.0))
                aovs[4] = 1.0
            else:
                aovs[3] = 1.0

        block.put(pos if box_filter else sample_pos, aovs, active)

    def render(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        seed: int,
        spp: int,
        develop: bool,
        evaluate: bool,
    ) -> mi.TensorXf:
        m_stop = False
        m_samples_per_pass = -1

        film = sensor.film()
        film_size = film.crop_size()
        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        sampler = sensor.sampler()

        if spp > 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()

        spp_per_pass = spp if m_samples_per_pass == -1 else min(m_samples_per_pass, spp)

        if spp % spp_per_pass != 0:
            raise Exception(
                "sample_count (%d) must be a multiple of spp_per_pass (%d).",
                spp,
                spp_per_pass,
            )

        n_passes = spp / spp_per_pass

        n_channels = film.prepare(self.aov_names())

        result = mi.TensorXf()

        if dr.is_jit_v(mi.Float):
            if n_passes > 1 and not evaluate:
                evaluate = True

            wavefront_size = film_size.x * film_size.y * spp_per_pass

            sampler.set_samples_per_wavefront(spp_per_pass)

            sampler.seed(seed, int(wavefront_size))

            block: mi.ImageBlock = film.create_block()
            block.set_offset(film.crop_offset())

            block.set_coalesce(block.coalesce() & spp_per_pass >= 4)

            idx = dr.arange(mi.UInt32, wavefront_size)
            idx //= spp_per_pass

            pos = mi.Vector2f()
            pos.y = idx // film_size.x
            pos.x = idx % film_size.x

            if film.sample_border():
                pos -= film.rfilter().border_size()

            pos += film.crop_offset()

            diff_scale_factor = dr.rsqrt(spp)

            aovs = [mi.Float32] * n_channels

            for i in range(int(n_passes)):
                self.render_sample(
                    scene, sensor, sampler, block, aovs, pos, diff_scale_factor
                )
                if n_passes > 1:
                    sampler.advance()
                    sampler.schedule_state()
                    dr.eval(block.tensor())

            film.put_block(block)

            if develop:
                result = film.develop()
                dr.schedule(result)
            else:
                film.schedule_storage()

            if evaluate:
                dr.eval()

            return result

    def sample_emitter(
        self,
        scene: mi.Scene,
        si: mi.SurfaceInteraction3f,
        bsdf_ctx: mi.BSDFContext,
        f: mi.Spectrum,
        sampler: mi.Sampler,
        active: mi.Bool,
    ):
        bsdf: mi.BSDF = si.bsdf()
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.All)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )
        active_em &= ds.pdf != 0.0

        with dr.resume_grad():
            ds.d = dr.normalize(ds.p - si.p)
            em_val = scene.eval_emitter_direction(si, ds, active_em)
            em_weight = dr.select((ds.pdf != 0), em_val / ds.pdf, 0)
            dr.disable_grad(ds.d)

        wo = si.to_local(ds.d)
        bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

        return f * bsdf_val * em_weight * mis_em

    def direct_emission(
        self,
        scene: mi.Scene,
        si: mi.SurfaceInteraction3f,
        prev_si: mi.SurfaceInteraction3f,
        prev_bsdf_pdf: mi.Float,
        prev_bsdf_delta: mi.Bool,
        f: mi.Spectrum,
    ):
        ds = mi.DirectionSample3f(scene, si, prev_si)
        em_pdf = scene.eval_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

        with dr.resume_grad():
            L = f * ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf

        return L

    @dr.syntax
    def sample_Lo(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        si: mi.SurfaceInteraction3f,
        medium: mi.Medium = None,
        max_depth: int = 128,
        rr_depth: int = 8,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool]:
        # ----------------------- Primary emission ------------------------

        L = si.emitter(scene, active).eval(si, active)

        # --------------------- Configure loop state ----------------------
        f = mi.Spectrum(1.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        # Variables caching information from the previous bounce
        bsdf_ctx = mi.BSDFContext()
        active = mi.Bool(active)
        active &= depth < max_depth

        while dr.hint(active, max_iterations=-1):
            # ---------------------- Emitter sampling ----------------------
            L[active] += self.sample_emitter(scene, si, bsdf_ctx, f, sampler, active)

            # ---------------------- BSDF sampling ----------------------
            bsdf = si.bsdf()

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            if dr.grad_enabled(ray):
                ray = dr.detach(ray)

                wo = si.to_local(ray.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)
                bsdf_weight[bsdf_pdf > 0.0] = bsdf_val / dr.detach(bsdf_pdf)

            # ------ Update loop variables based on current interaction ------

            f[active] *= bsdf_weight
            eta[active] *= bsdf_sample.eta

            prev_si = dr.detach(si)

            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
            # Call bsdf with ray to compute uv partials
            si.bsdf(ray)

            active &= si.is_valid()

            # ---------------------- Direct emission ----------------------
            L[active] += self.direct_emission(
                scene,
                si,
                prev_si,
                bsdf_sample.pdf,
                mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta),
                f,
            )

            # -------------------- Stopping criterion ---------------------

            depth[active] += 1

            fmax = dr.max(f)

            rr_prob = dr.minimum(fmax * dr.sqr(eta), 0.95)
            rr_active = depth >= rr_depth
            rr_continue = sampler.next_1d() < rr_prob

            f[rr_active] *= dr.rcp(dr.detach(rr_prob))

            active = (
                active
                & (~rr_active | rr_continue)
                & (fmax != 0.0)
                & (depth < max_depth)
            )

        return L, (depth != 0)

    def sample(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool]:
        # Get primary intersection
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        # Call bsdf with ray to compute uv partials
        si.bsdf(ray)

        return self.sample_Lo(
            scene,
            sampler,
            si,
            medium,
            max_depth=self.max_depth - 1,
            rr_depth=self.rr_depth,
            active=active,
        )


mi.register_integrator("mypath", lambda props: Path(props))

if __name__ == "__main__":
    scene = mi.cornell_box()
    scene = mi.load_dict(scene)

    mypath = mi.load_dict(
        {
            "type": "mypath",
            "max_depth": 16,
            "rr_depth": 4,
        }
    )

    path = mi.load_dict(
        {
            "type": "path",
            "max_depth": 16,
            "rr_depth": 4,
        }
    )

    dr.kernel_history_clear()
    img = mi.render(scene, integrator=mypath, spp=128)
    kernels = dr.kernel_history()
    optix_kernels = [
        kernel
        for kernel in kernels
        if "uses_optix" in kernel and kernel["uses_optix"] == 1
    ]
    print(f"My Path: {optix_kernels}")
    print("")

    dr.kernel_history_clear()
    ref = mi.render(scene, integrator=path, spp=128)
    kernels = dr.kernel_history()
    optix_kernels = [
        kernel
        for kernel in kernels
        if "uses_optix" in kernel and kernel["uses_optix"] == 1
    ]
    print(f"Default Path: {optix_kernels}")

    diff = dr.abs(img - ref)

    mse = dr.mean(dr.sqr(diff), axis=None)
    print(f"{mse=}")

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    ax[0].imshow(mi.util.convert_to_bitmap(img))
    ax[0].set_title("img")
    ax[1].imshow(mi.util.convert_to_bitmap(ref))
    ax[1].set_title("ref")
    ax[2].imshow(mi.util.convert_to_bitmap(diff))
    ax[2].set_title("diff")
    plt.show()
