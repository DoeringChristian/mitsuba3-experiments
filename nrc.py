# [markdown]
# # Neural Radiance Caching

# [markdown]
# ## Imports

# %%

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from mitsuba.ad.integrators.common import mis_weight

# %%


class NRCIntegrator(mi.SamplingIntegrator):
    def __init__(self) -> None:
        props = mi.Properties("integrator")
        super().__init__(props)
        self.max_depth = 10

    @dr.syntax
    def next_segment(
        self,
        si: mi.SurfaceInteraction3f,
        scene: mi.Scene,
        sampler: mi.Sampler,
        c: mi.Float,
        a0: mi.Float,
        active: bool | mi.Bool = True,
    ):

        L = mi.Spectrum(0.0)
        f = mi.Spectrum(1.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(1)
        spread = mi.Float(0.0)
        active = mi.Bool(active)
        bsdf_ctx = mi.BSDFContext()

        while dr.hint(active, mode="evaluated"):
            bsdf = si.bsdf()

            # Emitter Sampling

            active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active
            )
            active_em &= ds.pdf != 0.0

            wo = si.to_local(ds.d)

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sampler.next_1d(), sampler.next_2d()
            )

            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf))
            L[active_em] += f * bsdf_val * em_weight * mis_em

            f *= bsdf_weight
            eta *= bsdf_sample.eta

            # -------------------- Stopping criterion ---------------------
            # Stop the path segment if

            a = dr.square(spread)
            active &= a < (c * a0)

            # ---------------------- Direct emission ----------------------
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si2: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

            bsdf_delta: mi.Bool = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta
            )

            ds = mi.DirectionSample3f(scene, si=si2, ref=si)
            em_pdf = scene.pdf_emitter_direction(si, ds, ~bsdf_delta)
            mis_bsdf = mis_weight(bsdf_sample.pdf, em_pdf)

            L[active] += f * ds.emitter.eval(si2, bsdf_sample.pdf > 0.0) * mis_bsdf

            # -------------------- Stopping criterion ---------------------

            # Equation 3
            spread += dr.sqrt(
                dr.squared_norm(si2.p - si.p) / (bsdf_sample.pdf * dr.abs(si2.wi.z))
            )

            # ----------------------- Loop State Update ------------------------
            si = dr.detach(si2, True)
            depth[active] += 1

            active &= depth < self.max_depth
            active &= si.is_valid()

        return L

    @dr.syntax
    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool]:

        L = mi.Spectrum(0.0)

        # ----------------------- Primary sample ------------------------
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        # L += si.emitter(scene, active).eval(si, active)
        active &= si.is_valid()

        a0 = dr.squared_norm(ray.o - si.p) / (dr.four_pi * dr.abs(si.wi.z))

        L += self.next_segment(si, scene, sampler, 0.01, a0, active)

        return L, si.is_valid(), []


# %%

scene = mi.cornell_box()
# scene["white"] = {
#     "type": "principled",
#     "specular": 1.0,
#     "roughness": 0.0,
# }
scene = mi.load_dict(scene)
integrator = NRCIntegrator()

img = mi.render(scene, integrator=integrator, spp=1)
mi.util.write_bitmap("out/nrc/img.exr", img)
