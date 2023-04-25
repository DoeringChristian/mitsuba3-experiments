from pssmlt import Path, Pssmlt, drjitstruct, mis_weight
import mitsuba as mi
import drjit as dr


@drjitstruct
class PathVert:
    wo: mi.Vector3f
    emitter_sample: mi.Point2f


class PssmltPath(Pssmlt):
    def __init__(self, props: mi.Properties) -> None:
        self.path_type = PathVert
        super().__init__(props)

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        path: Path,
        initialize: bool,
        wavefront_size: int,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:
        # if initialize:
        #     self.emitter_offset = Path(wavefront_size, self.max_depth, mi.Vector2f)
        # path = Path(PathVert, len(ray.d.x), self.max_depth)

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        f = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(scene.environment() is not None)

        # Variables caching information from the previous bounce
        prev_si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
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
                valid_ray,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            ),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si = scene.ray_intersect(ray)  # TODO: not necesarry in first interaction

            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = mi.Float(0.0)

            em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            L = dr.fma(
                f,
                ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf,
                L,
            )

            active_next = ((depth + 1) < self.max_depth) & si.is_valid()

            bsdf: mi.BSDF = si.bsdf(ray)

            # ------ Evaluate BSDF * cos(theta) and sample direction -------

            # sample1 = sampler.next_1d()
            # sample2 = sampler.next_2d()

            # bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
            #     bsdf_ctx, si, wo, sample1, sample2
            # )

            # ---------------------- BSDF sampling ----------------------

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d()
            )

            vert: PathVert = self.mutate(
                self.path[depth], bsdf_sample.wo, sampler.next_2d()
            )

            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, vert.wo, active)

            vert.wo[bsdf_pdf <= 0.0] = bsdf_sample.wo
            bsdf_weight[bsdf_pdf > 0.0] = bsdf_val / bsdf_pdf

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(vert.wo))

            # ---------------------- Emitter sampling ----------------------

            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, em_weight = scene.sample_emitter_direction(
                si, vert.emitter_sample, True, active_em
            )

            wo = si.to_local(ds.d)

            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo)

            # --------------- Emitter sampling contribution ----------------

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mi_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            L[active_em] = dr.fma(f, bsdf_val * em_weight * mi_em, L)

            # ------ Update loop variables based on current interaction ------

            path[depth] = vert

            f *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= (
                active
                & si.is_valid()
                & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            )

            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(f)

            rr_prop = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prop

            f[rr_active] *= dr.rcp(rr_prop)

            active = (
                active_next & (~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )

        return L

    def mutate(self, old: PathVert, wo: mi.Vector3f, sample1: mi.Point2f) -> PathVert:
        vert = PathVert()
        vert.wo = dr.normalize(old.wo + wo)
        vert.emitter_sample = dr.clamp(
            mi.warp.square_to_std_normal(sample1) * 0.2 + old.emitter_sample, 0.0, 1.0
        )

        return vert


mi.register_integrator("pssmlt", lambda props: PssmltPath(props))
