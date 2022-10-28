from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

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


class PathVert:
    ...


class Path:
    idx: mi.UInt32

    def __init__(self, n_rays: int, max_depth: int, dtype=PathVert):
        self.n_rays = n_rays
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, n_rays)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.n_rays))

    def __setitem__(self, depth: mi.UInt32, value):
        dr.scatter(self.vertices, value, depth * self.n_rays + self.idx)

    # Return vertex at depth
    @overload
    def __getitem__(self, depth: mi.UInt32) -> PathVert:
        ...

    # Return a vertex at (depth, ray_index)
    @overload
    def __getitem__(self, idx: tuple[mi.UInt32, mi.UInt32]) -> PathVert:
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


class Pssmlt(mi.SamplingIntegrator):
    path: Path
    L: mi.Color3f
    sample_count = 0

    def __init__(self, props: mi.Properties) -> None:
        self.max_depth = props.get("max_depth", def_value=16)
        self.rr_depth = props.get("rr_depth", def_value=4)
        super().__init__(props)

    def sample(
        self: mi.SamplingIntegrator,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ):
        if self.sample_count == 0:
            self.path = Path(len(ray.d.x), self.max_depth, dtype=mi.Vector3f)
            self.L = mi.Color3f(0)

        L, path, valid = self.sample_rest(scene, sampler, ray, medium, active)
        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0.0, 1.0)
        u = sampler.next_1d()

        self.L = dr.select(u <= a, L, self.L)
        u = dr.tile(u, self.max_depth)
        a = dr.tile(a, self.max_depth)
        self.path.vertices = dr.select(u <= a, path.vertices, self.path.vertices)

        self.sample_count += 1
        return self.L, valid, []

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:
        path = Path(len(ray.d.x), self.max_depth, dtype=mi.Vector3f)

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
            wo += self.path[depth]
            wo = dr.normalize(wo)
            path[depth] = wo

            ray = si.spawn_ray(si.to_world(wo))

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

        return L, path, dr.neq(depth, 0)


mi.register_integrator("pssmlt", lambda props: Pssmlt(props))

scene = mi.cornell_box()
integrator = mi.load_dict(
    {
        "type": "pssmlt",
        "max_depth": 8,
        "rr_depth": 2,
    }
)
# scene["sensor"]["film"]["width"] = 128
# scene["sensor"]["film"]["height"] = 128
scene["sphere"] = {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.translate([0.335, -0.7, 0.38]).scale(0.3),
    "bsdf": {"type": "dielectric"},
}
scene["blocking"] = {
    "type": "cube",
    "to_world": mi.ScalarTransform4f.translate([0.0, 0.4, 0.0]).scale(0.3),
}
del scene["small-box"]
print(f"{scene=}")
scene = mi.load_dict(scene)

img = None
for i in range(50):
    print(f"{i=}")
    img = mi.render(scene, integrator=integrator, seed=i)
    mi.util.write_bitmap(f"out/{i}.png", img)

plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
