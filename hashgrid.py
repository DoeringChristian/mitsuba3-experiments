import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def hash(p: mi.Point3u | mi.Point3f, hash_size: int):
    if isinstance(p, mi.Point3f):
        p = mi.Point3u(mi.UInt(p.x), mi.UInt(p.y), mi.UInt(p.z))
        return hash(p, hash_size)
    return ((p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791)) % hash_size


class HashGrid:
    def __init__(
        self, sample: mi.Point3f, resolution: int, n_cells: None | int = None
    ) -> None:
        """
        Constructs a 3D Hash Grid with the samples inserted.

        It uses the hash function from the pbrt-v3 SPPM implementaiton (https://github.com/mmp/pbrt-v3/blob/master/src/integrators/sppm.cpp)

        @param sample: Samples to insert into the Hash Grid
        @param resolution: The number of grid cells in each dimension
        """
        n_samples = dr.shape(sample)[-1]
        if n_cells is None:
            n_cells = n_samples
        self.n_cells = n_cells
        self.n_samples = n_samples
        self.resolution = resolution
        self.bbmin = dr.minimum(
            dr.min(sample.x), dr.minimum(dr.min(sample.y), dr.min(sample.z))
        )
        self.bbmax = dr.maximum(
            dr.max(sample.x), dr.maximum(dr.max(sample.y), dr.max(sample.z))
        )
        self.bbmax = dr.maximum(
            dr.max(sample.x), dr.maximum(dr.max(sample.y), dr.max(sample.z))
        )

        from prefix_sum import prefix_sum

        cell = self.cell_idx(sample)

        cell_size = dr.zeros(mi.UInt, n_cells)
        index_in_cell = mi.UInt(0)
        processing = dr.zeros(mi.UInt, n_cells)
        queued = mi.Bool(True)

        while dr.any(queued):
            dr.scatter(processing, dr.arange(mi.UInt, n_samples), cell, active=queued)
            selected = (
                dr.eq(
                    dr.gather(mi.UInt, processing, cell, queued),
                    dr.arange(mi.UInt, n_samples),
                )
                & queued
            )
            index_in_cell[selected] = dr.gather(mi.UInt, cell_size, cell, selected)
            dr.scatter(cell_size, index_in_cell + 1, cell, selected)
            queued &= ~selected

        first_cell = dr.eq(dr.arange(mi.UInt, n_cells), 0)
        cell_offset = prefix_sum(cell_size)
        cell_offset = dr.select(
            first_cell,
            0,
            dr.gather(
                mi.UInt,
                cell_offset,
                dr.arange(mi.UInt, n_cells) - 1,
                ~first_cell,
            ),
        )
        self.cell_offset = cell_offset
        self.cell_size = cell_size
        self.sample_idx = dr.zeros(mi.UInt, n_samples)
        dr.scatter(
            self.sample_idx,
            dr.arange(mi.UInt, n_samples),
            dr.gather(mi.UInt, cell_offset, cell) + index_in_cell,
        )

    def cell_idx(self, p: mi.Point3f):
        return hash(
            (p - self.bbmin) / (self.bbmax - self.bbmin) * self.resolution,
            self.n_cells,
        )


if __name__ == "__main__":
    x = mi.Float(0, 0.1, 0.6, 1)
    y = mi.Float(0, 0.1, 0.6, 1)
    z = mi.Float(0, 0.1, 0.6, 1)

    grid = HashGrid(mi.Point3f(x, y, z), 2, 2)
