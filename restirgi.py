import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb")


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class PathIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth: int = props.get("max_depth", 8)
        self.rr_depth: int = props.get("rr_depth", 2)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:
        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        throughput = mi.Spectrum(1.0)
        result = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(scene.environment() is not None)

        # Variables caching information from the previous bounce
        prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "Path Tracer",
            state=lambda: (
                sampler,
                ray,
                throughput,
                result,
                eta,
                depth,
                valid_ray,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            ),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, mi.RayFlags.All, dr.eq(depth, 0)
            )

            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = mi.Float(0.0)

            em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            result = dr.fma(
                throughput,
                ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf,
                result,
            )

            active_next = ((depth + 1) < self.max_depth) & si.is_valid()

            bsdf: mi.BSDF = si.bsdf(ray)

            # ---------------------- Emitter sampling ----------------------

            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em
            )

            wo = si.to_local(ds.d)

            # ------ Evaluate BSDF * cos(theta) and sample direction -------

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sample1, sample2
            )

            # --------------- Emitter sampling contribution ----------------

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mi_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            result[active_em] = dr.fma(throughput, bsdf_val * em_weight * mi_em, result)

            # ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= (
                active
                & si.is_valid()
                & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            )

            prev_si = mi.Interaction3f(si)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(throughput)

            rr_prop = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prop

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active_next & (~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )

        return dr.select(valid_ray, result, 0.0), valid_ray, []


mi.register_integrator("path_test", lambda props: PathIntegrator(props))

if __name__ == "__main__":
    with dr.suspend_grad():
        scene = mi.load_dict(mi.cornell_box())

        integrator = mi.load_dict(
            {
                "type": "path_test",
            }
        )

        img = mi.render(scene, integrator=integrator, spp=128)

        plt.imshow(mi.util.convert_to_bitmap(img))
        plt.show()
