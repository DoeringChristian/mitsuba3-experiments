import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    dr.set_flag(dr.JitFlag.LoopRecord, False)


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
    def expand_sample(
        self, sample: mi.Point3f, radius: mi.Float
    ) -> tuple[mi.UInt, mi.UInt]:
        initial_sample_size = dr.shape(sample)[-1]
        pmin = self.to_grid(sample - mi.Vector3f(radius))
        pmax = self.to_grid(sample + mi.Vector3f(radius)) + 1

        grid_size: mi.Vector3u = pmax - pmin
        bins_per_grid = grid_size.x * grid_size.y * grid_size.z
        sample_size = dr.sum(bins_per_grid)[0]

        dr.eval(bins_per_grid)
        grid_offset = cumsum(bins_per_grid)
        print(f"{grid_offset=}")
        print(f"{bins_per_grid=}")

        sample_idx = dr.zeros(mi.UInt, sample_size)
        sample_h = dr.zeros(mi.UInt, sample_size)
        inside = dr.zeros(mi.Bool, sample_size)

        idx = mi.UInt(0)
        dr.set_flag(dr.JitFlag.LoopRecord, False)
        loop = mi.Loop("Bin Size", lambda: (idx,))

        while loop(idx < bins_per_grid):
            z = idx // grid_size.x * grid_size.y
            y = idx % grid_size.z // grid_size.x
            x = idx % grid_size.z % grid_size.y
            p = mi.Point3u(x, y, z)
            p = p - grid_size // 2 + pmin
            h = hash(p, sample_size)
            # print(f"{sample_idx=}")
            # print(f"{grid_offset + idx=}")
            # dr.scatter(
            #     inside,
            #     (p > 0) & (p < self.resolution),
            #     grid_offset + idx,
            #     idx < bins_per_grid,
            # )
            dr.scatter(
                sample_idx,
                dr.arange(mi.UInt, initial_sample_size),
                grid_offset + idx,
                idx < bins_per_grid,
            )
            # print(f"{grid_offset+idx=}")
            dr.scatter(
                sample_h,
                h,
                grid_offset + idx,
            )
            idx += 1

        print(f"{sample_size=}")
        print(f"{dr.count(inside)=}")
        print(f"{sample_idx=}")

        idx = dr.compress(inside)
        sample_h = dr.gather(mi.UInt, sample_h, idx)
        sample_idx = dr.gather(mi.UInt, sample_idx, idx)

        print(f"{sample_h=}")
        return sample_idx, sample_h

    def __init__(self, sample: mi.Point3f, radius: mi.Float, resolution: int) -> None:
        """
        Constructs a 3D Hash Grid with the samples inserted.

        It uses the hash function from the pbrt-v3 SPPM implementaiton (https://github.com/mmp/pbrt-v3/blob/master/src/integrators/sppm.cpp)

        @param sample: Samples to insert into the Hash Grid
        @param resolution: The number of grid cells in each dimension
        """
        # First expand samples

        # hash_size = dr.shape(sample)[-1]
        self.resolution = resolution
        self.bbmin = mi.Point3f(dr.min(sample.x), dr.min(sample.y), dr.min(sample.z))
        self.bbmax = (
            mi.Point3f(dr.max(sample.x), dr.max(sample.y), dr.max(sample.z)) + 0.0001
        )

        ref_sample_idx, h = self.expand_sample(sample, radius)
        sample_size = dr.shape(ref_sample_idx)[-1]

        """
        In order to calculate the offset for every bin we first calculate the
        size of every bin using the `scatter_reduce` function.
        The size is written into the `bin_size` array at the hash position `h`.
        Afterwards we calculate the cumulative sum in order to get an offset for
        every bin.
        Now querying the `bin_offset` array at position `h` gets the offset for the
        bin corresponding to that hash.
        """
        bin_size = dr.zeros(mi.UInt, sample_size)
        dr.scatter_reduce(dr.ReduceOp.Add, bin_size, 1, h)
        dr.eval(bin_size)
        bin_offset = cumsum(bin_size)  # This represents

        sample_bin_offset = dr.gather(mi.UInt, bin_offset, h)

        sample_idx = dr.zeros(mi.UInt, sample_size)
        sample_cell_cap = dr.gather(mi.UInt, bin_size, h)
        active_sample = dr.full(mi.Bool, True, sample_size)

        """
        In this loop we iterate through all cells in a bin and from high to low insert
        the index of the sample into the `sample_idx` array.
        In order to not insert indices twice we need to 'deactivate' samples that have 
        already been inserted.
        To do so we need to get the last inserted index which is only possible after
        calling `dr.eval` on `sample_idx`.
        Therefore the loop cannot be a Dr.Jit loop.
        """

        depth = mi.UInt(0)
        max_depth = dr.max(bin_size)[0]
        loop_record = dr.flag(dr.JitFlag.LoopRecord)
        dr.set_flag(dr.JitFlag.LoopRecord, False)

        loop = mi.Loop("Fill bins", lambda: (depth))

        while loop(depth < max_depth):
            dr.scatter_reduce(
                dr.ReduceOp.Max,
                sample_idx,
                ref_sample_idx,
                # dr.arange(mi.UInt, hash_size),
                depth + sample_bin_offset,
                (depth < sample_cell_cap) & active_sample,
            )
            dr.eval(sample_idx)

            selected_sample = dr.gather(
                mi.UInt, sample_idx, depth + sample_bin_offset, depth < sample_cell_cap
            )
            is_selected_sample = dr.eq(selected_sample, ref_sample_idx)
            active_sample &= ~is_selected_sample

        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)

        self.__bin_size = bin_size
        self.__bin_offset = bin_offset
        self.__sample_idx = sample_idx
        self.__sample = sample

    def to_grid(self, p: mi.Point2f) -> mi.Point3u:
        p_grid = dr.clamp(
            mi.Point3u((p - self.bbmin) / (self.bbmax - self.bbmin) * self.resolution),
            mi.Point3u(0),
            mi.Point3u(self.resolution),
        )
        return p_grid

    def hash(self, sample: mi.Point2f):
        return hash(
            (sample - self.bbmin) / (self.bbmax - self.bbmin) * self.resolution,
            self.sample_size,
        )


class SPPMIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

    def sample_visible_point(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        sample_pos: mi.Point2f,
    ) -> tuple[mi.SurfaceInteraction3f, mi.Spectrum]:
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))
        max_depth = 6
        β = mi.Spectrum(1.0)
        depth = mi.UInt(0)
        active = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()
        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop("Camera Tracing", lambda: (depth, active, β, ray, si))
        loop.set_max_iterations(max_depth)

        while loop(active):
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

            bsdf: mi.BSDF = si.bsdf(ray)
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )

            active &= si.is_valid()
            active &= ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Smooth)
            active &= depth < max_depth

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            β[active] *= bsdf_weight
            depth += 1

        return si, β

    def render(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        seed: int,
        spp: int,
        develop: bool,
        evaluate: bool,
    ) -> mi.TensorXf:
        film = sensor.film()
        film_size = film.crop_size()

        wavefront_size = film_size.x * film_size.y

        sampler = sensor.sampler()
        sampler.set_sample_count(1)
        sampler.set_samples_per_wavefront(1)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)
        pos = mi.Vector2u()
        pos.y = idx // film_size.x
        pos.x = dr.fma(-film_size.x, pos.y, idx)

        sample_pos = (mi.Point2f(pos) + sampler.next_2d()) / mi.Point2f(
            film.crop_size()
        )

        # Sample visible points:

        vp_si, vp_β = self.sample_visible_point(scene, sensor, sampler, sample_pos)

        dr.eval(vp_si, vp_β)

        grid = HashGrid(vp_si.p, 100)
        print(f"{vp_β=}")
        print(f"{dr.count(vp_si.is_valid())=}")

        ...


if __name__ == "__main__":
    sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler.seed(2, 10)
    gird = HashGrid(
        mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d()),
        sampler.next_1d() * 0.01,
        100,
    )
    # scene: mi.Scene = mi.load_dict(mi.cornell_box())
    #
    # integrator = SPPMIntegrator(mi.Properties())
    #
    # integrator.render(scene, scene.sensors()[0], 0, 1, True, True)
    # mi.render(scene, integrator=integrator)
