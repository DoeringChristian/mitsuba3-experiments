import mitsuba as mi
import drjit as dr
from tqdm import tqdm
from drjitstruct import drjitstruct

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def J_rcp(q: "RestirSample", r: "RestirSample") -> mi.Float:
    """
    Calculate the Reciprocal of the absolute of the Jacobian determinant.
    J_rcp = |J_{q\\rightarrow r}|^{-1} // Equation 11 from paper
    """
    w_qq = q.x_v - q.x_s
    w_qq_len = dr.norm(w_qq)
    w_qq /= w_qq_len
    cos_psi_q = dr.clamp(dr.dot(w_qq, q.n_s), 0, 1)

    w_qr = r.x_v - q.x_s
    w_qr_len = dr.norm(w_qr)
    w_qr /= w_qr_len
    cos_psi_r = dr.clamp(dr.dot(w_qr, q.n_s), 0, 1)

    div = dr.abs(cos_psi_r) * dr.sqr(w_qq_len)
    return dr.select(div > 0, dr.abs(cos_psi_q) * dr.sqr(w_qr_len) / div, 0.0)


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def p_hat(f):
    return dr.norm(f)


class ReuseSet:
    def __init__(self):
        self.M = []
        self.active = []
        self.p = []
        self.n = []

    def put(self, M: mi.UInt, pos: mi.Vector3f, n: mi.Vector3f, active: mi.Bool = True):
        self.M.append(M)
        self.p.append(pos)
        self.n.append(n)
        self.active.append(mi.Bool(active))

    def __len__(self) -> int:
        assert len(self.M) == len(self.p) == len(self.active) == len(self.n)
        return len(self.M)


@drjitstruct
class RestirSample:
    x_v: mi.Vector3f
    n_v: mi.Vector3f
    x_s: mi.Vector3f
    n_s: mi.Vector3f

    L_o: mi.Color3f
    p_q: mi.Float
    valid: mi.Bool


@drjitstruct
class RestirReservoir:
    z: RestirSample
    w: mi.Float
    W: mi.Float
    M: mi.UInt

    def update(
        self,
        sampler: mi.Sampler,
        snew: RestirSample,
        wnew: mi.Float,
        active: mi.Bool = True,
    ):
        active = mi.Bool(active)
        if dr.shape(active)[-1] == 1:
            dr.make_opaque(active)

        self.w += dr.select(active, wnew, 0)
        self.M += dr.select(active, 1, 0)
        self.z: RestirSample = dr.select(
            active & (sampler.next_1d() < wnew / self.w), snew, self.z
        )

    def merge(
        self, sampler: mi.Sampler, r: "RestirReservoir", p, active: mi.Bool = True
    ):
        active = mi.Bool(active)
        M0 = mi.UInt(self.M)
        self.update(sampler, r.z, p * r.W * r.M, active)
        self.M = dr.select(active, M0 + r.M, M0)


class RestirIntegrator(mi.SamplingIntegrator):
    dist_threshold = 0.1
    angle_threshold = 25 * dr.pi / 180

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth: int = props.get("max_depth", 8)
        self.rr_depth: int = props.get("rr_depth", 2)
        self.bias_correction = props.get("bias_correction", True)
        self.jacobian = props.get("jacobian", True)
        self.bsdf_sampling = props.get("bsdf_sampling", True)
        self.max_M_temporal = props.get("max_M_temporal", None)
        self.max_M_spatial = props.get("max_M_spatial", None)
        self.initial_search_radius = props.get("initial_search_radius", 10.0)
        self.minimal_search_radius = props.get("minimal_search_radius", 3.00)
        self.n = 0
        self.film_size: None | mi.Vector2u = None

    def to_idx(self, pos: mi.Vector2u) -> mi.UInt:
        pos = dr.clamp(mi.Point2u(pos), mi.Point2u(0), self.film_size)
        assert self.film_size is not None
        return (pos.y * self.film_size.x + pos.x) * self.spp + self.sample_offset

    def similar(self, s1: RestirSample, s2: RestirSample) -> mi.Bool:
        dist = dr.norm(s1.x_v - s2.x_v)
        similar = dist < self.dist_threshold
        similar &= dr.dot(s1.n_v, s2.n_v) > dr.cos(self.angle_threshold)

        return similar

    def render(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        seed: int = 0,
        spp: int = 1,
        develop: bool = True,
        evaluate: bool = True,
    ):
        film = sensor.film()

        film_size = film.crop_size()

        if self.film_size is None:
            self.film_size = film_size

        wavefront_size = film_size.x * film_size.y * spp

        sampler = sensor.sampler()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)

        pos = mi.Vector2u()
        pos.x = idx // spp % film_size.x
        pos.y = idx // film_size.x // spp
        self.sample_offset = idx % spp
        self.spp = spp

        sample_pos = (mi.Point2f(pos) + sampler.next_2d()) / mi.Point2f(
            film.crop_size()
        )

        if self.n == 0:
            self.temporal_reservoir: RestirReservoir = dr.zeros(
                RestirReservoir, wavefront_size
            )
            self.spatial_reservoir: RestirReservoir = dr.zeros(
                RestirReservoir, wavefront_size
            )
            self.search_radius = dr.full(
                mi.Float, self.initial_search_radius, wavefront_size
            )

            self.prev_sensor: mi.Sensor = mi.load_dict({"type": "perspective"})
            mi.traverse(self.prev_sensor).update(mi.traverse(sensor))

        self.sample_initial(scene, sampler, sensor, sample_pos)
        dr.eval(self.sample)
        if self.n == 0:
            self.prev_sample = self.sample
        self.temporal_resampling(sampler, mi.Vector2f(pos))
        dr.eval(self.temporal_reservoir)
        self.spatial_resampling(scene, sampler, pos)
        dr.eval(self.spatial_reservoir, self.search_radius)

        res = self.render_final()

        film.prepare(self.aov_names())
        block: mi.ImageBlock = film.create_block()

        aovs = [res.x, res.y, res.z, mi.Float(1)]

        block.put(pos, aovs)

        film.put_block(block)

        img = film.develop()
        dr.eval(img)

        # Update n, prev_sensor and prev_sample
        self.n += 1
        mi.traverse(self.prev_sensor).update(mi.traverse(sensor))
        self.prev_sample = self.sample

        return img

    def render_final(self) -> tuple[mi.Color3f, mi.Color3f, mi.Color3f]:
        assert self.film_size is not None
        R = self.spatial_reservoir
        S = R.z

        si = self.si_v
        bsdf: mi.BSDF = self.si_v.bsdf()
        β = bsdf.eval(mi.BSDFContext(), si, si.to_local(dr.normalize(S.x_s - si.p)))

        result = β * S.L_o * R.W + self.emittance

        return result

    def spatial_resampling(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        pos: mi.Vector2u,
    ):
        Rs = self.spatial_reservoir

        """
        Create a new reservoir to merge the spatial reservoirs into.
        This is neccesary so we can clamp M.
        """
        Rnew: RestirReservoir = dr.zeros(RestirReservoir)
        Q = ReuseSet()

        q: RestirSample = self.sample

        Rnew.merge(sampler, Rs, p_hat(Rs.z.L_o))
        # Q.put(Rs.M, Rs.z.x_v, Rs.z.n_v, self.similar(q, Rs.z))

        max_iter = dr.select(Rs.M < self.max_M_spatial / 2, 9, 3)

        any_reused = dr.full(mi.Bool, False, len(pos.x))

        for s in range(9):
            active = s < max_iter

            offset = (
                mi.warp.square_to_uniform_disk(sampler.next_2d()) * self.search_radius
            )
            p = dr.clamp(pos + mi.Vector2i(offset), mi.Point2u(0), self.film_size)

            qn: RestirSample = dr.gather(RestirSample, self.sample, self.to_idx(p))

            active &= self.similar(qn, q)

            Rn: RestirReservoir = dr.gather(
                RestirReservoir, self.temporal_reservoir, self.to_idx(p), active
            )  # l.9

            si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
            si.p = q.x_v
            si.n = q.n_v
            shadowed = scene.ray_test(si.spawn_ray_to(Rn.z.x_s), active)

            phat = dr.select(
                ~active | shadowed,
                0,
                p_hat(Rn.z.L_o) * (J_rcp(Rn.z, q) if self.jacobian else 1.0),
            )  # l.11 - 13

            Rnew.merge(sampler, Rn, phat, active)

            Q.put(Rn.M, Rn.z.x_v, Rn.z.n_v, active)

            any_reused |= active

        phat = p_hat(Rnew.z.L_o)
        if self.bias_correction:
            Z = mi.UInt(Rs.M)
            for i in range(len(Q)):
                active = Q.active[i]

                si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
                si.p = Rnew.z.x_s
                si.n = Rnew.z.n_s
                ray = si.spawn_ray_to(Q.p[i])

                # active &= dr.dot(ray.d, Q.n[i]) < 0
                active &= ~scene.ray_test(ray, active)

                Z += dr.select(active, Q.M[i], 0)

            Rnew.W = dr.select(Z * phat > 0, Rnew.w / (Z * phat), 0.0)
        else:
            Rnew.W = dr.select(phat * Rnew.M > 0, Rnew.w / (Rnew.M * phat), 0)

        # Decrease search radius:
        self.search_radius = dr.maximum(
            dr.select(any_reused, self.search_radius, self.search_radius / 2),
            self.minimal_search_radius,
        )

        if self.max_M_spatial is not None:
            Rnew.M = dr.minimum(Rnew.M, self.max_M_spatial)

        self.spatial_reservoir = Rnew

    def temporal_resampling(
        self,
        sampler: mi.Sampler,
        pos: mi.Vector2f,
    ):
        S = self.sample

        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si.p = S.x_v
        ds, _ = self.prev_sensor.sample_direction(
            si, mi.Point2f(0.0)
        )  # type: tuple[mi.DirectionSample3f, mi.Color3f]
        ds: mi.DirectionSample3f = ds

        valid = ds.pdf > 0

        Sprev: RestirSample = dr.gather(
            RestirSample, self.prev_sample, self.to_idx(mi.Point2u(ds.uv)), valid
        )

        valid &= self.similar(S, Sprev)

        R = dr.select(valid, self.temporal_reservoir, dr.zeros(RestirReservoir))

        """
        Create a new reservoir to update with the new sample and merge the old temporal reservoir into.
        This is necesarry to limit the samples in the old temporal reservoir.
        """
        Rnew: RestirReservoir = dr.zeros(RestirReservoir)

        phat = p_hat(S.L_o)
        w = dr.select(S.p_q > 0, phat / S.p_q, 0.0)  # Weight for new sample
        Rnew.update(sampler, S, w)  # Add new sample to Rnew

        Rnew.merge(sampler, R, p_hat(R.z.L_o))  # add min(R.M, CLAMP) samples from R

        phat = p_hat(Rnew.z.L_o)
        Rnew.W = dr.select(
            phat * Rnew.M > 0, Rnew.w / (Rnew.M * phat), 0
        )  # Update Contribution Weight W in Rnew

        if self.max_M_temporal is not None:
            Rnew.M = dr.minimum(Rnew.M, self.max_M_temporal)

        self.temporal_reservoir = Rnew

    def sample_initial(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        sensor: mi.Sensor,
        # pos: mi.Vector2u,
        sample_pos: mi.Point2f,
    ) -> RestirSample:
        S = RestirSample()
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf()

        ds = mi.DirectionSample3f(scene, si, dr.zeros(mi.SurfaceInteraction3f))
        emitter: mi.Emitter = ds.emitter
        self.emittance = emitter.eval(si)

        S.x_v = si.p
        S.n_v = si.n
        S.valid = si.is_valid()
        self.si_v = si

        if self.bsdf_sampling:
            bsdf_sample, bsdf_weight = bsdf.sample(
                mi.BSDFContext(), si, sampler.next_1d(), sampler.next_2d()
            )

            wo = bsdf_sample.wo
            pdf = bsdf_sample.pdf
        else:
            wo = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
            pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)

        S.p_q = pdf

        ray = si.spawn_ray(si.to_world(wo))

        S.L_o = self.sample_ray(scene, sampler, ray)

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        S.x_s = si.p
        S.n_s = si.n

        self.sample = S

    def sample_ray(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        active: bool = True,
    ) -> mi.Color3f:
        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        throughput = mi.Spectrum(1.0)
        result = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(scene.environment() is not None)

        # Variables caching information from the previous bounce
        prev_si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "Path Tracer",
            state=lambda: (
                sampler,
                ray,
                throughput,
                result,
                eta,
                depth,
                valid_ray,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            ),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si = scene.ray_intersect(ray)  # TODO: not necesarry in first interaction

            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = mi.Float(0.0)

            em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            result = dr.fma(
                throughput,
                ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf,
                result,
            )

            active_next = ((depth + 1) < self.max_depth) & si.is_valid()

            bsdf: mi.BSDF = si.bsdf(ray)

            # ---------------------- Emitter sampling ----------------------

            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em
            )

            wo = si.to_local(ds.d)

            # ------ Evaluate BSDF * cos(theta) and sample direction -------

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sample1, sample2
            )

            # --------------- Emitter sampling contribution ----------------

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mi_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            result[active_em] = dr.fma(throughput, bsdf_val * em_weight * mi_em, result)

            # ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= (
                active
                & si.is_valid()
                & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            )

            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(throughput)

            rr_prop = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prop

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active_next & (~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )

        return dr.select(valid_ray, result, 0.0)


mi.register_integrator("restirgi", lambda props: RestirIntegrator(props))

if __name__ == "__main__":
    with dr.suspend_grad():
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = 1024
        scene["sensor"]["film"]["height"] = 1024
        scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
        scene: mi.Scene = mi.load_dict(scene)
        # scene: mi.Scene = mi.load_file("./data/scenes/living-room-3/scene.xml")
        # scene: mi.Scene = mi.load_file("data/scenes/staircase/scene.xml")
        # scene: mi.Scene = mi.load_file("data/scenes/shadow-mask/scene.xml")
        # scene: mi.Scene = mi.load_file("data/scenes/dining-room/scene.xml")

        print("Rendering Reference Image:")
        ref = mi.render(scene, spp=256)
        mi.util.write_bitmap("out/ref.jpg", ref)

        integrator: RestirIntegrator = mi.load_dict(
            {
                "type": "restirgi",
                "jacobian": False,
                "bias_correction": False,
                "bsdf_sampling": True,
                "max_M_spatial": 500,
                "max_M_temporal": 30,
                "initial_search_radius": 10,
            }
        )

        print("ReSTIRGI:")
        for i in tqdm(range(200)):
            img = mi.render(scene, integrator=integrator, seed=i, spp=1)

            mi.util.write_bitmap(f"out/{i}.jpg", img)
