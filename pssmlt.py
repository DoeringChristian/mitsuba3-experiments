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


class Path:
    def __init__(self, dtype, wavefront_size: int, max_depth: int):
        self.wavefront_size = wavefront_size
        self.max_depth = max_depth
        # self.idx = dr.arange(mi.UInt32, wavefront_size)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.wavefront_size))

    def __setitem__(self, depth: mi.UInt32, value):
        dr.scatter(
            self.vertices,
            value,
            depth * self.wavefront_size + dr.arange(mi.UInt32, self.wavefront_size),
        )

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
                self.dtype,
                self.vertices,
                idx * self.wavefront_size + dr.arange(mi.UInt32, self.wavefront_size),
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
    # sample_count = 0
    cumulative_weight: mi.Float32
    path_type: ...

    def __init__(self, props: mi.Properties) -> None:
        self.max_depth = props.get("max_depth", def_value=16)
        self.rr_depth = props.get("rr_depth", def_value=4)
        super().__init__(props)

    def reset(self):
        ...

    def render_sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        sensor: mi.Sensor,
        block: mi.ImageBlock,
        pos: mi.Vector2u,
        large_step: mi.Bool,
        agregate: mi.Bool,
    ):
        large_step = mi.Bool(large_step)
        agregate = mi.Bool(agregate)
        film = sensor.film()

        proposed_offset = self.mutate_offset(self.offset, sampler.next_2d(), large_step)

        sample_pos = (mi.Point2f(pos) + proposed_offset) / mi.Point2f(film.crop_size())
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))

        L = (
            self.sample(scene, sampler, ray, self.proposed, large_step=large_step)
            * ray_weight
        )
        dr.schedule(self.proposed.vertices)

        a = dr.clamp(mi.luminance(L) / mi.luminance(self.L), 0.0, 1.0)
        u = sampler.next_1d()

        accept = u < a
        proposed_weight = a
        current_weight = 1.0 - a
        self.cumulative_weight[accept] = proposed_weight
        self.cumulative_weight[~accept] += current_weight
        dr.schedule(self.cumulative_weight)

        # self.offset = dr.select(u < a, offset, self.offset)
        self.offset[accept] = proposed_offset
        dr.schedule(self.offset)

        # self.L = dr.select(accept, L, self.L)
        self.L[accept] = L
        dr.schedule(self.L)

        accept = dr.tile(accept, self.max_depth)
        self.path.vertices = dr.select(
            accept, self.proposed.vertices, self.path.vertices
        )
        dr.schedule(self.path.vertices)

        res = self.L / self.cumulative_weight
        dr.schedule(self.cumulative_weight)

        aovs = [res.x, res.y, res.z, mi.Float(1.0)]
        block.put(pos, aovs, active=agregate)

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

        wavefront_size = film_size.x * film_size.y * spp
        print(f"{wavefront_size=}")

        sampler = sensor.sampler()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)
        idx //= spp

        pos = mi.Vector2u()
        pos.y = idx // film_size.x
        pos.x = dr.fma(-film_size.x, pos.y, idx)

        # Initialize State:
        self.path = Path(self.path_type, wavefront_size, self.max_depth)
        self.proposed = Path(self.path_type, wavefront_size, self.max_depth)
        self.offset = mi.Vector2f(0.5)
        self.L = mi.Color3f(0)
        self.cumulative_weight = mi.Float32(0.0)

        film.prepare(self.aov_names())

        block: mi.ImageBlock = film.create_block()

        reset_interval = 50
        bootstrapping_count = 40
        for i in range(200):
            large_step = i % reset_interval == 0
            agregate = i % reset_interval > bootstrapping_count
            print(f"Iteration: {i}")
            print(f"{large_step=}")
            print(f"{agregate=}")

            self.render_sample(scene, sampler, sensor, block, pos, large_step, agregate)

            sampler.advance()
            sampler.schedule_state()
            dr.eval(block.tensor())

        film.put_block(block)

        img = film.develop()
        dr.schedule(img)
        dr.eval()

        # self.sample_count += 1
        return img

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        path: Path,
        large_step: mi.Bool,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, Path]:
        ...

    def init_path(self, wavefront_size):
        ...

    def mutate_offset(self, x_old: mi.Vector2f, xnew: mi.Vector2f, large_step: mi.Bool):
        large_step = mi.Bool(large_step)
        return dr.select(
            large_step,
            xnew,
            dr.clamp(
                mi.warp.square_to_std_normal(xnew) * dr.sqrt(0.1) + x_old,
                0.0,
                1.0,
            ),
        )
