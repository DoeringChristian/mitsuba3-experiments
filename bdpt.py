import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from typing import Type, TypeVar, overload

mi.set_variant("llvm_ad_rgb")
# dr.set_log_level(dr.LogLevel.Debug)

T = TypeVar("T")


class Path:
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
