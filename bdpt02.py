import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from typing import Generic, Type, TypeVar, overload
from dataclasses import dataclass

mi.set_variant("llvm_ad_rgb")
# dr.set_log_level(dr.LogLevel.Debug)


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


T = TypeVar("T")


@dataclass
class Vertex:
    def __init__(self) -> None:
        self.p: mi.Point3f = dr.zeros(mi.Point3f)
        self.f: mi.Color3f = dr.zeros(mi.Color3f)
        self.L: mi.Color3f = dr.zeros(mi.Color3f)
        self.wi: mi.Vector3f = dr.zeros(mi.Vector3f)

    DRJIT_STRUCT = {
        "p": mi.Point3f,
        "f": mi.Color3f,
        "L": mi.Color3f,
        "wi": mi.Vector3f,
    }


class Path(Generic[T]):
    idx: mi.UInt32

    def __init__(self, dtype: Type[T], n_rays: int, max_depth: int):
        self.n_rays = n_rays
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, n_rays)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.n_rays))

    def __setitem__(self, depth: mi.UInt32, value: T):
        dr.scatter(self.vertices, value, depth * self.n_rays + self.idx)

    # Return vertex at depth
    @overload
    def __getitem__(self, depth: mi.UInt32) -> T:
        ...

    # Return a vertex at (depth, ray_index)
    @overload
    def __getitem__(self, idx: tuple[mi.UInt32, mi.UInt32]) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, mi.UInt32):
            return dr.gather(self.dtype, self.vertices, idx * self.n_rays + self.idx)
        if (
            isinstance(idx, tuple)
            and isinstance(idx[0], mi.UInt32)
            and isinstance(idx[1], mi.UInt32)
        ):
            return dr.gather(self.dtype, self.vertices, idx[0] * self.n_rays + idx[1])


class BDPTIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        self.max_depth = int(props.get("max_depth", def_value=16))
        self.rr_depth = int(props.get("rr_depth", def_value=4))
        super().__init__(props)

    def record_light_path(
        self, scene: mi.Scene, sampler: mi.Sampler, active: bool = True
    ) -> Path[Vertex]:
        wavefront_size = sampler.wavefront_size()
        path = Path(Vertex, wavefront_size, self.max_depth)

        ray, ray_weight, emitter = scene.sample_emitter_ray(
            0.0, sampler.next_1d(), sampler.next_2d(), sampler.next_2d(), active
        )

        bsdf_ctx = mi.BSDFContext()

        depth = mi.UInt32(0)
        f = mi.Color3f(1.0)
        L = mi.Color3f(ray_weight)
        active = mi.Bool(active)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(
            name="Record Light",
            state=lambda: (sampler, ray, depth, f, L, active, prev_si),
        )
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray,
            )

            bsdf: mi.BSDF = si.bsdf()

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            bsdf_sample, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
            )
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            vertex = Vertex()
            vertex.f = f
            vertex.L = L
            vertex.p = si.p
            vertex.wi = si.to_world(si.wi)
            path[depth] = vertex

            f *= bsdf_val
            L = f * L + Le

            prev_si = dr.detach(si, True)

            active = active_next
            depth += 1

        return path

    def record_camera_path(
        self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, active: bool = True
    ) -> Path[Vertex]:
        wavefront_size = sampler.wavefront_size()
        path = Path(Vertex, wavefront_size, self.max_depth)

        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)
        f = mi.Color3f(1.0)
        L = mi.Color3f(0.0)
        active = mi.Bool(active)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(
            name="Record View",
            state=lambda: (sampler, ray, depth, f, L, active, prev_si),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0)
            )

            bsdf: mi.BSDF = si.bsdf(ray)

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = f * ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            bsdf_sample, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
            )

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            L = L + Le

            vertex = Vertex()
            vertex.f = f
            vertex.L = L
            vertex.p = si.p
            vertex.wi = si.to_world(si.wi)
            path[depth] = vertex

            f *= bsdf_val

            prev_si = dr.detach(si, True)

            active = active_next
            depth += 1

        return path

    def connect_s2t(
        self,
        scene: mi.Scene,
        s: mi.UInt32,
        t: mi.UInt32,
        s_path: Path[Vertex],
        t_path: Path[Vertex],
    ) -> tuple[mi.Color3f, mi.Color3f]:
        """
        Perform connection between vectex s and t.
        Returns bsdf weight at vertex s and radiance emitted from s in direction of t.

        s0   s1   s2   t2   t1   t0
        o -- o -- o .. o -- o -- o

        first ray is cast from t2 to s2 to test if the points are visible to each other and to get surface interaction at s2.
        Then we use wi (s2 -> s1) to calculate the bsdf weight.
        """
        s_p = s_path[s].p
        t_p = t_path[t].p

        t2s_dir = dr.normalize(s_p - t_p)

        t2s_ray = mi.Ray3f(t_p, t2s_dir)

        active = scene.ray_test(mi.Ray3f(t2s_ray, dr.norm(s_p - t_p)))

        si: mi.SurfaceInteraction3f = scene.ray_intersect(t2s_ray, active)

        bsdf: mi.BSDF = si.bsdf()

        wo = si.to_local(s_path[s].wi)
        weight, pdf = bsdf.eval_pdf(mi.BSDFContext(), si, wo, active)
        weight = dr.select(pdf > 0, weight / pdf, 0.0)
        weight = dr.select(active, weight, 0.0)

        emitter: mi.Emitter = si.emitter(scene, active)
        Le = emitter.eval(si, active)

        return weight, Le

    def connect_bdpt(
        self,
        scene: mi.Scene,
        s: mi.UInt32,
        t: mi.UInt32,
        camera_path: Path[Vertex],
        light_path: Path[Vertex],
    ) -> mi.Color3f:

        camera_weight, camera_Le = self.connect_s2t(
            scene, s, t, camera_path, light_path
        )
        light_weight, light_Le = self.connect_s2t(scene, t, s, light_path, camera_path)

        L = (
            camera_path[s].L
            + camera_path[s].f * camera_weight * light_Le
            + camera_path[s].f * camera_weight * light_weight * light_path[t].L
        )
        return L

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Bool, list]:
        camera_path = self.record_camera_path(scene, sampler, ray, active)
        light_path = self.record_light_path(scene, sampler, active)
        f = camera_path[mi.UInt32(1)].f
        L = self.connect_bdpt(
            scene, mi.UInt32(1), mi.UInt32(1), camera_path, light_path
        )
        return L, mi.Bool(True), []


mi.register_integrator("bdpt", lambda props: BDPTIntegrator(props))

scene = mi.cornell_box()
scene = mi.load_dict(scene)
integrator = mi.load_dict(
    {
        "type": "bdpt",
        "max_depth": 16,
        "rr_depth": 2,
    }
)

img = mi.render(scene, integrator=integrator)
plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
