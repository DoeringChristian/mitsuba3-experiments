from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass


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


class MLTSampler(mi.Sampler):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        self.prng = mi.PCG32()

    def next_1d(self, active: bool = True) -> float:
        return super().next_1d(active)

    def next_2d(self, active: bool = True) -> mi.Point2f:
        return super().next_2d(active)

    def advance(self) -> None:
        return super().advance()

    def seed(self, seed: int, wavefront_size: int = 4294967295) -> None:
        super().seed(seed, wavefront_size)

        self.mutation_idx = dr.arange(mi.UInt32, self.wavefront_size())

        idx = dr.arange(mi.UInt32, self.wavefront_size())
        tmp = dr.opaque(seed)

        v0, v1 = mi.sample_tea_32(idx, tmp)
        self.prng.seed(1, v0, v1)


class Pssmlt(mi.SamplingIntegrator):
    wo: Path
    L: mi.Color3f
    sample_count = 0
    nee = True
    cumulative_weight: mi.Float32

    def __init__(self, props: mi.Properties) -> None:
        self.max_depth = props.get("max_depth", def_value=16)
        self.rr_depth = props.get("rr_depth", def_value=4)
        self.nee = props.get("nee", def_value=True)
        super().__init__(props)

    def reset(self):
        self.sample_count = 0

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ):
        if self.sample_count == 0:
            self.wo = Path(len(ray.d.x), self.max_depth, dtype=mi.Vector3f)
            self.L = mi.Color3f(0)
            self.cumulative_weight = mi.Float32(0.0)

        L, path, valid = self.sample_rest(scene, sampler, ray, medium, active)
        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0.0, 1.0)
        u = sampler.next_1d()

        proposed_weight = a
        current_weight = 1.0 - a
        self.cumulative_weight = dr.select(
            u < a, proposed_weight, self.cumulative_weight + current_weight
        )

        self.L = dr.select(u < a, L, self.L)
        u = dr.tile(u, self.max_depth)
        a = dr.tile(a, self.max_depth)
        self.wo.vertices = dr.select(u < a, path.vertices, self.wo.vertices)
        dr.schedule(self.L)
        dr.schedule(self.wo.vertices)
        dr.eval()

        self.sample_count += 1
        return self.L / self.cumulative_weight, valid, []

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:
        ...
