import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb")


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def drjitstruct(cls):
    annotations = cls.__dict__.get("__annotations__", {})
    drjit_struct = {}
    for name, type in annotations.items():
        drjit_struct[name] = type
    cls.DRJIT_STRUCT = drjit_struct
    return cls


@drjitstruct
class Sample:
    xv: mi.Point3f
    nv: mi.Vector3f
    xs: mi.Point3f
    ns: mi.Vector3f
    Lo: mi.Color3f
    rand: mi.Point3f


class Reservoir:
    z: Sample
    w = mi.Float32(0)
    M = mi.Float32(0)
    W = mi.Float32(0)

    def update(self, sampler: mi.Sampler, s_new: Sample, w_new: mi.Float32):
        self.w += w_new
        self.M += 1
        self.z = dr.select(sampler.next_1d() < w_new / self.w, s_new, self.z)

    def merge(self, sampler: mi.Sampler, r: "Reservoir", p_hat: mi.Float32):
        M0 = mi.Float32(self.M)
        self.update(sampler, r.z, p_hat * r.W * r.M)
        self.M = M0 + r.M


class Restir(mi.SamplingIntegrator):
    reservoir = Reservoir()

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

        medium = sensor.medium()

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

    def sample(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:

        # --------------------- Configure loop state ----------------------
        ray = mi.Ray3f(ray)
        f = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        active = mi.Bool(active)
        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "Path Tracer",
            state=lambda: (
                sampler,
                ray,
                f,
                L,
                eta,
                depth,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            ),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            with dr.resume_grad():
                si: mi.SurfaceInteraction3f = scene.ray_intersect(
                    ray, mi.RayFlags.All, coherent=mi.Bool(False)
                )

            # ---------------------- Direct emission ----------------------
            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = scene.eval_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            # L = dr.fma(f, ds.emitter.eval(si, prev_bsdf_pdf > 0.) * mis_bsdf, L)
            with dr.resume_grad():
                # Le = f * mis_bsdf * ds.emitter.eval(si)
                L = dr.fma(f, ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf, L)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # ---------------------- Emitter sampling ----------------------

            bsdf: mi.BSDF = si.bsdf(ray)
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.All)

            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em
            )
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad():
                ds.d = dr.normalize(ds.p - si.p)
                em_val = scene.eval_emitter_direction(si, ds, active_em)
                em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)
                dr.disable_grad(ds.d)

            wo = si.to_local(ds.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            L[active_em] = dr.fma(f, bsdf_val * em_weight * mis_em, L)

            # ---------------------- BSDF sampling ----------------------
            s1 = sampler.next_1d()
            s2 = sampler.next_2d()

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, s1, s2, active_next)
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            if dr.grad_enabled(ray):
                ray = dr.detach(ray)

                wo = si.to_local(ray.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)
                bsdf_weight[bsdf_pdf > 0.0] = bsdf_val / dr.detach(bsdf_pdf)

            # ------ Update loop variables based on current interaction ------

            f *= bsdf_weight
            eta *= bsdf_sample.eta

            prev_si = dr.detach(si)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            fmax = dr.max(f)

            rr_prob = dr.minimum(fmax * dr.sqr(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob

            active = active_next & (~rr_active | rr_continue) & dr.neq(fmax, 0.0)

        return L, dr.neq(depth, 0)


mi.register_integrator("restir", lambda props: Restir(props))

scene = mi.cornell_box()
scene = mi.load_dict(scene)
integrator = mi.load_dict(
    {
        "type": "restir",
        "max_depth": 16,
        "rr_depth": 2,
    }
)

img = mi.render(scene, integrator=integrator)
plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
