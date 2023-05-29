import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def hash(p: mi.Point3u | mi.Point3f, hash_size: int):
    if isinstance(p, mi.Point3f):
        p = mi.Point3u(mi.UInt(p.x), mi.UInt(p.y), mi.UInt(p.z))
        return hash(p, hash_size)
    return ((p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791)) % hash_size


def cumsum(src: mi.UInt | mi.Float):
    N = dr.shape(src)[-1]
    idx = dr.arange(mi.UInt, N)
    dst = dr.zeros(type(src), N)
    depth = mi.UInt(0)

    loop = mi.Loop("cumsum", lambda: (idx, dst, depth))

    loop.set_max_iterations(N)

    while loop(depth < idx):
        dst += dr.gather(mi.UInt, src, depth, depth < idx)
        depth += 1

    return dst


class HashGrid:
    def __init__(
        self, sample: mi.Point3f, resolution: int, hash_size: None | int = None
    ) -> None:
        """
        Constructs a 3D Hash Grid with the samples inserted.

        It uses the hash function from the pbrt-v3 SPPM implementaiton (https://github.com/mmp/pbrt-v3/blob/master/src/integrators/sppm.cpp)

        @param sample: Samples to insert into the Hash Grid
        @param resolution: The number of grid cells in each dimension
        """
        n_samples = dr.shape(sample)[-1]
        if hash_size is None:
            self.hash_size = n_samples
        self.n_samples = n_samples
        self.resolution = resolution
        self.bbmin = mi.Point3f(dr.min(sample.x), dr.min(sample.y), dr.min(sample.z))
        self.bbmax = (
            mi.Point3f(dr.max(sample.x), dr.max(sample.y), dr.max(sample.z)) + 0.0001
        )

        h = self.bin_idx(sample)

        """
        In order to calculate the offset for every bin we first calculate the
        size of every bin using the `scatter_reduce` function.
        The size is written into the `bin_size` array at the hash position `h`.
        Afterwards we calculate the cumulative sum in order to get an offset for
        every bin.
        Now querying the `bin_offset` array at position `h` gets the offset for the
        bin corresponding to that hash.
        """
        bin_size = dr.zeros(mi.UInt, self.hash_size)
        dr.scatter_reduce(dr.ReduceOp.Add, bin_size, 1, h)
        dr.eval(bin_size)
        bin_offset = cumsum(bin_size)  # This represents

        sample_bin_offset = dr.gather(mi.UInt, bin_offset, h)

        sample_idx = dr.zeros(mi.UInt, n_samples)
        sample_cell_cap = dr.gather(mi.UInt, bin_size, h)
        active_sample = dr.full(mi.Bool, True, n_samples)

        loop_record = dr.flag(dr.JitFlag.LoopRecord)
        dr.set_flag(dr.JitFlag.LoopRecord, False)
        depth = mi.UInt(0)
        max_depth = dr.max(bin_size)[0]

        loop = mi.Loop("Fill Bins", lambda: (depth))
        loop.set_max_iterations(max_depth)

        """
        In this loop we iterate through all cells in a bin and from high to low insert
        the index of the sample into the `sample_idx` array.
        In order to not insert indices twice we need to 'deactivate' samples that have 
        already been inserted.
        To do so we need to get the last inserted index which is only possible after
        calling `dr.eval` on `sample_idx`.
        Therefore the loop cannot be a Dr.Jit loop.
        """
        while loop(depth < sample_cell_cap):
            prev_inserted = dr.gather(
                mi.UInt,
                sample_idx,
                depth + sample_bin_offset - 1,
                depth < sample_cell_cap & (depth > 0),
            )
            is_inserted = dr.eq(prev_inserted, dr.arange(mi.UInt, n_samples)) & (
                depth > 0
            )
            active_sample &= ~is_inserted

            dr.scatter_reduce(
                dr.ReduceOp.Max,
                sample_idx,
                dr.arange(mi.UInt, n_samples),
                depth + sample_bin_offset,
                (depth < sample_cell_cap) & active_sample,
            )
            depth += 1

        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)

        self.__bin_size = bin_size
        self.__bin_offset = bin_offset
        self.__sample_idx = sample_idx
        self.__sample = sample

    def bin_idx(self, p: mi.Point3f) -> mi.UInt:
        return hash(
            (p - self.bbmin) / (self.bbmax - self.bbmin) * self.resolution,
            self.hash_size,
        )

    def bin_offset(self, idx: mi.UInt) -> mi.UInt:
        return dr.gather(mi.UInt, self.__bin_offset, idx)

    def bin_size(self, idx: mi.UInt) -> mi.UInt:
        return dr.gather(mi.UInt, self.__bin_size, idx)

    def sample_idx(self, idx: mi.UInt) -> mi.UInt:
        return dr.gather(mi.UInt, self.__sample_idx, idx)


if __name__ == "__main__":
    N = 100

    sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler.seed(0, N)
    p = mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())

    grid = HashGrid(p, 2)

    idx = grid.bin_idx(mi.Point3f(0.1, 0.1, 0.1))

    print(f"{idx=}")
    print(f"{grid.bin_offset(idx)=}")
    print(f"{grid.bin_size(idx)=}")

    sample_idx = grid.sample_idx(
        grid.bin_offset(idx)[0] + dr.arange(mi.UInt, grid.bin_size(idx)[0])
    )

    print(f"{sample_idx=}")
    p = dr.gather(mi.Point3f, p, sample_idx)
    print(f"{p=}")
