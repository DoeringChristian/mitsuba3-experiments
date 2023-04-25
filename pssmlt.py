from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gc


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


# class PathVert:
#     ...


class Path:
    idx: mi.UInt32

    def __init__(self, dtype, wavefront_size: int, max_depth: int):
        self.wavefront_size = wavefront_size
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, wavefront_size)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.wavefront_size))

    def __setitem__(self, depth: mi.UInt32, value):
        dr.scatter(self.vertices, value, depth * self.wavefront_size + self.idx)

    # Return vertex at depth
    @overload
    def __getitem__(self, depth: mi.UInt32):
        ...

    # Return a vertex at (depth, ray_index)
    @overload
    def __getitem__(self, idx: tuple[mi.UInt32, mi.UInt32]):
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
    path_type: ...

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
            # self.init_path()
            self.path = Path(self.path_type, wavefront_size, self.max_depth)
            self.proposed = Path(self.path_type, wavefront_size, self.max_depth)
            # self.wo = Path(wavefront_size, self.max_depth, dtype=mi.Vector3f)
            self.L = mi.Color3f(0)
            self.cumulative_weight = mi.Float32(0.0)
            self.offset = mi.Vector2f(0.5)

        # dr.schedule(self.L)
        # dr.schedule(self.path.vertices)

        sampler = sensor.sampler()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)
        idx //= spp

        pos = mi.Vector2u()
        pos.y = idx // film_size.x
        pos.x = dr.fma(-film_size.x, pos.y, idx)

        offset = self.mutate_2d(self.offset, sampler.next_2d())

        sample_pos = (mi.Point2f(pos) + offset) / mi.Point2f(film.crop_size())
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))

        L = self.sample_rest(
            scene, sampler, ray, self.proposed, self.sample_count == 0, wavefront_size
        )
        dr.schedule(self.proposed.vertices)
        # dr.eval(L, self.proposed.vertices)
        # dr.eval(self.proposed.vertices)
        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0.0, 1.0)
        u = sampler.next_1d()

        accept = u < a
        proposed_weight = a
        current_weight = 1.0 - a
        self.cumulative_weight = dr.select(
            accept, proposed_weight, self.cumulative_weight + current_weight
        )
        dr.schedule(self.cumulative_weight)

        self.offset = dr.select(u < a, offset, self.offset)
        dr.schedule(self.offset)

        self.L = dr.select(accept, L, self.L)
        dr.schedule(self.L)

        accept = dr.tile(accept, self.max_depth)
        self.path.vertices = dr.select(
            accept, self.proposed.vertices, self.path.vertices
        )
        dr.schedule(self.path.vertices)

        res = self.L / self.cumulative_weight
        film.prepare(self.aov_names())

        block: mi.ImageBlock = film.create_block()
        aovs = [res.x, res.y, res.z, mi.Float(1.0)]
        block.put(pos, aovs)
        film.put_block(block)

        img = film.develop()
        dr.schedule(img)
        dr.eval()

        self.sample_count += 1
        return img

    def sample_rest(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        wavefront_size: int,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, Path]:
        ...

    def init_path(self, wavefront_size):
        ...

    def mutate_2d(self, x: mi.Vector2f, xnew: mi.Vector2f):
        return dr.clamp(x + xnew * 0.2, 0.0, 1.0)
