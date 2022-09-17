
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import math

mi.set_variant("llvm_ad_rgb")
from pathrecord import Path, drjitstruct  # noqa


@drjitstruct
class PVert:
    wo: mi.Vector3f

    def __init__(self, wo=mi.Vector3f()):
        self.wo = wo


class MltSampler:
    pass


class Simple(mi.SamplingIntegrator):
    path: Path = None

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")
        self.pass_count = 0

    def sample(self: mi.SamplingIntegrator, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
        bsdf_ctx = mi.BSDFContext()

        n_rays = dr.shape(ray.o)[1]

        if self.path is None:
            self.path = Path(n_rays, self.max_depth, dtype=PVert)

            depth = dr.zeros(mi.UInt32, shape=n_rays)
            loop = mi.Loop(name="Init Rand", state=lambda: (depth))
            while loop(depth < self.max_depth):
                self.path[depth] = PVert(
                    mi.warp.square_to_uniform_sphere(sampler.next_2d()))
                depth += 1

            dr.eval(self.path.vertices)

            self.L = mi.Spectrum(0.)

        path = Path(n_rays, self.max_depth, dtype=PVert)

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)
        f = mi.Spectrum(1.)
        L = mi.Spectrum(0.)
        active = mi.Bool(active)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, depth, f, L, active, prev_si))

        loop.set_max_iterations(self.max_depth)

        while loop(active):

            pvert: PVert = self.path[depth]
            wo = mi.warp.square_to_uniform_sphere(sampler.next_2d())
            wo = dr.normalize(pvert.wo + wo * 0.)

            path[depth] = PVert(wo)

            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            bsdf: mi.BSDF = si.bsdf(ray)

            # Direct emission

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = f * ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            bsdf_val = bsdf.eval(
                bsdf_ctx, si, wo, active_next)

            # Update loop variables

            ray = si.spawn_ray(si.to_world(wo))
            L = (L + Le)
            f *= bsdf_val * 4. * math.pi

            prev_si = dr.detach(si, True)

            # Stopping criterion (russian roulettte)

            active_next &= dr.neq(dr.max(f), 0)

            rr_prop = dr.maximum(f.x, dr.maximum(f.y, f.z))
            rr_prop[depth < self.rr_depth] = 1.
            f *= dr.rcp(rr_prop)
            active_next &= (sampler.next_1d() < rr_prop)

            active = active_next
            depth += 1

        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0., 1.)

        u = sampler.next_1d()

        self.L = dr.select(u <= a, L, self.L)

        u = dr.tile(u, self.max_depth)
        a = dr.tile(a, self.max_depth)

        self.path.vertices = dr.select(
            u <= a, path.vertices, self.path.vertices)

        self.pass_count += 1
        return (self.L, dr.neq(depth, 0), [])


mi.register_integrator("integrator", lambda props: Simple(props))

scene = mi.cornell_box()
scene['integrator']['type'] = 'integrator'
scene['integrator']['max_depth'] = 16
scene['integrator']['rr_depth'] = 2
scene['integrator']['samples_per_pass'] = 16
scene['sensor']['sampler']['sample_count'] = 16
scene['sensor']['film']['width'] = 1024
scene['sensor']['film']['height'] = 1024
scene['sphere'] = {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.translate([0.335, -0.7, 0.38]).scale(0.3),
    "bsdf": {
        "type": "dielectric"
    }
}
del scene['small-box']
scene = mi.load_dict(scene)

img = mi.render(scene)

plt.imshow(mi.util.convert_to_bitmap(img))
plt.axis("off")
plt.show()
