import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")


def mis_weight(pdf_a, pdf_b):
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class Simple(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = 2

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray_: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray_)
        depth = mi.UInt32(0)
        f = mi.Spectrum(1.)
        L = mi.Spectrum(0.)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, depth, f, L, active, prev_si))

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            bsdf: mi.BSDF = si.bsdf(ray)

            # Direct emission

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = f * ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # BSDF Sampling
            bsdf_smaple, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            # Update loop variables

            ray = si.spawn_ray(si.to_world(bsdf_smaple.wo))
            L = (L + Le)
            f *= bsdf_val

            prev_si = dr.detach(si, True)

            # Stopping criterion (russian roulettte)

            active_next &= dr.neq(dr.max(f), 0)

            rr_prop = dr.maximum(f.x, dr.maximum(f.y, f.z))
            rr_prop[depth < self.rr_depth] = 1.
            f *= dr.rcp(rr_prop)
            active_next &= (sampler.next_1d() < rr_prop)

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), [])


mi.register_integrator("integrator", lambda props: Simple(props))

scene = mi.load_file("./scenes/cbox.xml", spp=64,
                     res=1024, integrator="integrator", max_depth=10)

img = mi.render(scene)

plt.imshow(img ** (1. / 2.2))
plt.axis("off")
plt.show()
