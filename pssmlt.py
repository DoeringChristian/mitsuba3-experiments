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

    def __init__(self, wavefront_size: int, max_depth: int, dtype=PathVert):
        self.wavefront_size = wavefront_size
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, wavefront_size)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.wavefront_size))

    def __setitem__(self, depth: mi.UInt32, value):
        dr.scatter(self.vertices, value, depth * self.wavefront_size + self.idx)

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
            return dr.gather(
                self.dtype, self.vertices, idx * self.wavefront_size + self.idx
            )
        if (
            isinstance(idx, tuple)
            and isinstance(idx[0], mi.UInt32)
            and isinstance(idx[1], mi.UInt32)
        ):
            return dr.gather(
                self.dtype, self.vertices, idx[0] * self.wavefront_size + idx[1]
            )


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
    offset: mi.Vector2f
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

    def render(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        seed: int = 0,
        spp: int = 1,
        develop: bool = True,
        evaluate: bool = True,
    ) -> mi.TensorXf:
        film = sensor.film()

        film_size = film.crop_size()

        # if self.film_size is None:
        #     self.film_size = film_size

        wavefront_size = film_size.x * film_size.y * spp
        print(f"{wavefront_size=}")

        if self.sample_count == 0:
            self.wo = Path(wavefront_size, self.max_depth, dtype=mi.Vector3f)
            self.L = mi.Color3f(0)
            self.cumulative_weight = mi.Float32(0.0)
            self.offset = mi.Vector2f(0.5)

        sampler = sensor.sampler()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)

        pos = mi.Vector2u()
        pos.y = idx // film_size.x
        pos.x = dr.fma(-film_size.x, pos.y, idx)

        offset = self.mutate_2d(self.offset, sampler.next_2d())

        sample_pos = (mi.Point2f(pos) + offset) / mi.Point2f(film.crop_size())
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))

        L, path, valid = self.sample_rest(scene, sampler, ray)
        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0.0, 1.0)
        u = sampler.next_1d()

        proposed_weight = a
        current_weight = 1.0 - a
        self.cumulative_weight = dr.select(
            u < a, proposed_weight, self.cumulative_weight + current_weight
        )

        self.offset = dr.select(u < a, offset, self.offset)

        self.L = dr.select(u < a, L, self.L)
        u = dr.tile(u, self.max_depth)
        a = dr.tile(a, self.max_depth)
        self.wo.vertices = dr.select(u < a, path.vertices, self.wo.vertices)

        res = self.L / self.cumulative_weight
        film.prepare(self.aov_names())

        block: mi.ImageBlock = film.create_block()
        aovs = [res.x, res.y, res.z, mi.Float(1.0)]
        block.put(pos, aovs)
        film.put_block(block)

        img = film.develop()
        dr.schedule(img)
        dr.schedule(self.L)
        dr.schedule(self.wo.vertices)
        dr.eval()

        self.sample_count += 1
        # return mi.TensorXf(
        #     dr.ravel(self.L / self.cumulative_weight),
        #     shape=[film_size.x, film_size.y, 3],
        # )
        return img

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> mi.Color3f:
        ...

    def mutate_2d(self, x: mi.Vector2f, xnew: mi.Vector2f):
        return dr.clamp(x + xnew * 0.2, 0.0, 1.0)

    def mutate_3d(self, x: mi.Vector3f, xnew: mi.Vector3f):
        return dr.normalize(x + xnew)
