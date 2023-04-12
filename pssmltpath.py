from pssmlt import Path, Pssmlt, mis_weight
import mitsuba as mi
import drjit as dr


class PssmltPath(Pssmlt):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:
        path_wo = Path(len(ray.d.x), self.max_depth, dtype=mi.Vector3f)

        # --------------------- Configure loop state ----------------------
        ray = mi.Ray3f(ray)
        f = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)
        bsdf_ctx = mi.BSDFContext()

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        active = mi.Bool(active)

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
            # mis_bsdf = 1.0

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

            # Pssmlt adjusting
            wo = bsdf_sample.wo
            wo += self.wo[depth]
            wo = dr.normalize(wo)

            # Reevaluate bsdf_weight after mutating wo
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)

            wo[bsdf_pdf <= 0.0] = bsdf_sample.wo
            bsdf_weight[bsdf_pdf > 0.0] = bsdf_val / bsdf_pdf

            path_wo[depth] = wo

            ray = si.spawn_ray(si.to_world(wo))

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

            f[rr_active] *= dr.rcp(dr.detach(rr_prob))

            active = active_next & (~rr_active | rr_continue) & dr.neq(fmax, 0.0)

        return L, path_wo, dr.neq(depth, 0)


mi.register_integrator("pssmlt", lambda props: PssmltPath(props))
