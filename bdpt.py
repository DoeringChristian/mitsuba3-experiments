

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")
# dr.set_log_level(dr.LogLevel.Debug)


def drjitstruct(cls):
    annotations = cls.__dict__.get('__annotations__', {})
    drjit_struct = {}
    for name, type in annotations.items():
        drjit_struct[name] = type
    cls.DRJIT_STRUCT = drjit_struct
    return cls


@drjitstruct
class PVert:
    f: mi.Spectrum
    L: mi.Spectrum
    p: mi.Point3f

    def __init__(self, f=mi.Spectrum(), L=mi.Spectrum(), p=mi.Point3f()):
        self.f = f
        self.L = L
        self.p = p


class Path:
    idx: mi.UInt32
    vertices: PVert

    def __init__(self, n_rays: int, max_depth: int):
        self.n_rays = n_rays
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, n_rays)

        self.vertices = dr.zeros(PVert, shape=(self.max_depth * self.n_rays))

    def __setitem__(self, depth: mi.UInt32, value: PVert):
        dr.scatter(self.vertices, value, depth * self.n_rays + self.idx)

    def __getitem__(self, depth: mi.UInt32) -> PVert:
        return dr.gather(PVert, self.vertices, depth * self.n_rays + self.idx)


class Simple(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")

    # record a path
    def record(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, active) -> Path:
        ray = mi.Ray3f(ray)
        bsdf_ctx = mi.BSDFContext()

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        n_rays = dr.shape(ray.o)[1]

        path = Path(n_rays, self.max_depth)
        depth = mi.UInt32(0)
        active = mi.Bool(active)

        loop = mi.Loop(name="Path Record", state=lambda: (
            sampler, ray, depth, active, prev_si))

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            bsdf: mi.BSDF = si.bsdf(ray)

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            L = ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            bsdf_sample, f = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            path[depth] = PVert(f, L, si.p)

            prev_si = dr.detach(si, True)

            active_next &= dr.neq(dr.max(f), 0)

            active = active_next
            depth += 1

        return path

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: mi.Bool = True):

        lray, lweight, emitter = scene.sample_emitter_ray(
            1., sampler.next_1d(), sampler.next_2d(), sampler.next_2d(), active)

        #lpath = self.record(scene, sampler, lray, True)

        vpath = self.record(scene, sampler, ray, True)

        dr.eval(vpath.vertices)

        depth = mi.UInt32(0)
        active = mi.Bool(active)
        L = mi.Spectrum(0.)
        f = mi.Spectrum(1.)
        loop = mi.Loop("Defferd lighting",
                       state=lambda: (sampler, depth, L, f, active))
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            vert = vpath[depth]
            L += vert.L * f
            f *= vert.f

            active = depth + 1 < self.max_depth
            depth += 1

        return (L, mi.Bool(True), [])


mi.register_integrator("integrator", lambda props: Simple(props))

scene = mi.cornell_box()
scene['integrator']['type'] = 'integrator'
scene['integrator']['max_depth'] = 16
scene['integrator']['rr_depth'] = 2
scene['sensor']['sampler']['sample_count'] = 1
scene['sensor']['film']['width'] = 1024
scene['sensor']['film']['height'] = 1024
scene = mi.load_dict(scene)

img = mi.render(scene)

plt.imshow(img ** (1. / 2.2))
plt.axis("off")
plt.show()
