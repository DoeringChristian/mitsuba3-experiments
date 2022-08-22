
import mitsuba as mi
from typing import overload
import drjit as dr


def drjitstruct(cls):
    annotations = cls.__dict__.get('__annotations__', {})
    drjit_struct = {}
    for name, type in annotations.items():
        drjit_struct[name] = type
    cls.DRJIT_STRUCT = drjit_struct
    return cls


# Need to record parameters to reconstruct surface intaraction
@drjitstruct
class PVert:
    f: mi.Spectrum
    L: mi.Spectrum
    i: mi.Interaction3f
    ps: mi.PositionSample3f

    def __init__(self, f=mi.Spectrum(), L=mi.Spectrum(), i=mi.Interaction3f(), ps=mi.PositionSample3f):
        self.f = f
        self.L = L
        self.i = i
        self.ps = ps


class Path:
    idx: mi.UInt32

    def __init__(self, n_rays: int, max_depth: int, dtype=PVert):
        self.n_rays = n_rays
        self.max_depth = max_depth
        self.idx = dr.arange(mi.UInt32, n_rays)
        self.dtype = dtype

        self.vertices = dr.zeros(dtype, shape=(self.max_depth * self.n_rays))

    def __setitem__(self, depth: mi.UInt32, value):
        dr.scatter(self.vertices, value, depth * self.n_rays + self.idx)

    # Return vertex at depth
    @overload
    def __getitem__(self, depth: mi.UInt32) -> PVert:
        ...

    # Return a vertex at (depth, ray_index)
    @overload
    def __getitem__(self, idx: (mi.UInt32, mi.UInt32)) -> PVert:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, mi.UInt32):
            return dr.gather(self.dtype, self.vertices, idx * self.n_rays + self.idx)
        if isinstance(idx, tuple) and isinstance(idx[0], mi.UInt32) and isinstance(idx[1], mi.UInt32):
            return dr.gather(self.dtype, self.vertices, idx[0] * self.n_rays + idx[1])
