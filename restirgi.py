import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def p_hat(f):
    return dr.norm(f)


class RestirSample:
    x_v: mi.Vector3f
    n_v: mi.Vector3f
    x_s: mi.Vector3f
    n_s: mi.Vector3f

    L_o: mi.Color3f
    p_q: mi.Float
    f: mi.Color3f

    DRJIT_STRUCT = {
        "x_v": mi.Vector3f,
        "n_v": mi.Vector3f,
        "x_s": mi.Vector3f,
        "n_s": mi.Vector3f,
        "L_o": mi.Color3f,
        "p_q": mi.Float,
        "f": mi.Color3f,
    }


class RestirReservoir:
    z: RestirSample
    w: mi.Float
    W: mi.Float
    M: mi.UInt

    DRJIT_STRUCT = {
        "z": RestirSample,
        "w": mi.Float,
        "W": mi.Float,
        "M": mi.UInt,
    }

    def update(self, sampler: mi.Sampler, snew: RestirSample, wnew: mi.Float):
        self.w += wnew
        self.M += 1
        self.z = dr.select(sampler.next_1d() < wnew / self.w, snew, self.z)


class PathIntegrator(mi.SamplingIntegrator):
    M_MAX = 500
    max_r = 10
    dist_threshold = 0.01
    angle_threshold = 25 * dr.pi / 180

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth: int = props.get("max_depth", 8)
        self.rr_depth: int = props.get("rr_depth", 2)
        self.n = 0
        self.film_size: None | mi.Vector2u = None

    def to_idx(self, pos: mi.Vector2u) -> mi.UInt:
        assert self.film_size is not None
        return pos.y * self.film_size.x + pos.x

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
        print(f"{wavefront_size=}")

        sampler = sensor.sampler()
        sampler.set_sample_count(spp)
        sampler.set_samples_per_wavefront(spp)
        sampler.seed(seed, wavefront_size)

        idx = dr.arange(mi.UInt, wavefront_size)

        pos = mi.Vector2u()
        pos.y = idx // film_size.x
        pos.x = dr.fma(-film_size.x, pos.y, idx)

        if self.n == 0:
            self.temporal_reservoir: RestirReservoir = dr.zeros(
                RestirReservoir, wavefront_size
            )
            self.spatial_reservoir: RestirReservoir = dr.zeros(
                RestirReservoir, wavefront_size
            )

        self.sample_initial(scene, sampler, sensor, pos)
        dr.schedule(self.initial_sample)
        dr.eval()
        self.temporal_resampling(sampler, idx)
        dr.schedule(self.temporal_reservoir)
        dr.eval()
        self.spatial_resampling(scene, sampler, pos)
        dr.schedule(self.spatial_reservoir)
        img = self.render_final("temporal")
        dr.schedule(img)
        dr.eval()

        return img

    def render_final(self, mode: str = "spatial") -> mi.TensorXf:
        assert self.film_size is not None
        if mode == "spatial":
            R = self.spatial_reservoir
            S = R.z
            wi = dr.normalize(S.x_s - S.x_v)
            color = S.f * S.L_o * R.W + self.emittance
        elif mode == "temporal":
            R = self.temporal_reservoir
            S = R.z
            wi = dr.normalize(S.x_s - S.x_v)
            color = S.f * S.L_o * R.W + self.emittance
        else:
            S = self.initial_sample
            color = S.f * S.L_o + self.emittance

        return mi.TensorXf(
            dr.ravel(color), shape=[self.film_size.x, self.film_size.y, 3]
        )

    def combine_reservoir(
        self,
        scene: mi.Scene,
        Rs: RestirReservoir,
        Rn: RestirReservoir,
        q: RestirSample,
        q_n: RestirSample,
        sampler: mi.Sampler,
    ):
        Rn_m = dr.minimum(Rn.M, self.M_MAX)
        Rn_hat = p_hat(Rn.z.L_o)
        shadowed = scene.ray_test(mi.Ray3f(q.x_v, dr.normalize(Rn.z.x_s - q.x_v)))

        w_qq = q.x_v - q.x_s
        w_qq_len = dr.norm(w_qq)
        w_qq /= w_qq_len
        w_rq = Rn.z.x_v - q.x_s
        w_rq_len = dr.norm(w_rq)
        w_rq /= w_rq_len
        qq = w_qq_len * w_qq_len
        rq = w_rq_len * w_rq_len
        div = rq * dr.abs(dr.dot(w_qq, q.n_s))
        j = dr.select(div == 0, 0, dr.abs(dr.dot(w_rq, q.n_s)) * qq / div)
        Rn_hat = dr.select(j == 0 | shadowed, 0, Rn_hat / j)

        factor = Rn_hat * Rn_m * Rn.W
        Rs.update(sampler, Rn.z, factor)

    def spatial_resampling(
        self, scene: mi.Scene, sampler: mi.Sampler, pos: mi.Vector2u
    ):
        R_s = self.spatial_reservoir

        max_iter = dr.select(R_s.M < self.M_MAX / 2, 9, 3)

        q = self.initial_sample

        q_n = RestirSample()
        R_n = RestirReservoir()

        Q = [
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
            mi.Vector3f(),
        ]
        Q_h = [
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
            mi.UInt(),
        ]
        Q_active = [
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
            mi.Bool(),
        ]

        Z = R_s.M
        sum = R_s.M
        for i in range(9):
            offset = mi.warp.square_to_uniform_disk(sampler.next_2d()) * self.max_r
            p = dr.clamp(pos + mi.Vector2i(offset), mi.Point2u(0), self.film_size)

            q_n: RestirSample = dr.gather(
                RestirSample, self.initial_sample, self.to_idx(p)
            )

            dist = dr.dot(q_n.x_v - q.x_v, q_n.x_v - q.x_v)
            active = dist > self.dist_threshold | (
                dr.dot(q_n.n_v, q.n_v) < dr.cos(self.angle_threshold)
            )

            R_n = dr.gather(
                RestirReservoir, self.temporal_reservoir, self.to_idx(p), active
            )

            self.combine_reservoir(scene, R_s, R_n, q, q_n, sampler)

            Q_h[i] = R_n.M
            Q[i] = q_n.x_s
            sum += R_n.M

        phat_val = p_hat(R_s.z.L_o)
        for i in range(len(Q)):
            shadowed = scene.ray_test(
                mi.Ray3f(R_s.z.x_v, dr.normalize(Q[i] - R_s.z.x_v))
            )
            Z += dr.select(~shadowed & (phat_val > 0.0), Q_h[i], 0)

        R_s.M = dr.minimum(sum, self.M_MAX)
        R_s.W = Z * dr.select(phat_val == 0, 0, R_s.w / (Z * phat_val))
        self.spatial_reservoir = R_s

    def temporal_resampling(
        self,
        sampler: mi.Sampler,
        idx: mi.UInt,
    ):
        S = self.initial_sample
        R = self.temporal_reservoir

        w = p_hat(S.L_o) / S.p_q
        R.update(sampler, S, w)
        phat = p_hat(R.z.L_o)
        R.W = dr.select(phat == 0, 0, R.w / (R.M * phat))

        self.temporal_reservoir = R

    def sample_initial(
        self, scene: mi.Scene, sampler: mi.Sampler, sensor: mi.Sensor, pos: mi.Vector2u
    ) -> RestirSample:
        film = sensor.film()

        S = RestirSample()

        sample_pos = (mi.Point2f(pos) + sampler.next_2d()) / mi.Point2f(
            film.crop_size()
        )
        ray, ray_weight = sensor.sample_ray(0.0, 0.0, sample_pos, mi.Point2f(0.5))

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf()

        ds = mi.DirectionSample3f(scene, si, dr.zeros(mi.SurfaceInteraction3f))
        emitter: mi.Emitter = ds.emitter
        self.emittance = emitter.eval(si)

        S.x_v = si.p
        S.n_v = si.n

        bsdf_sample, bsdf_weight = bsdf.sample(
            mi.BSDFContext(), si, sampler.next_1d(), sampler.next_2d()
        )

        S.p_q = bsdf_sample.pdf
        S.f = bsdf_weight

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        S.x_s = si.p
        S.n_s = si.n

        S.L_o = self.sample_outgoing(scene, sampler, si, ray)

        self.initial_sample = S

    def sample_outgoing(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        si: mi.SurfaceInteraction3f,
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


mi.register_integrator("path_test", lambda props: PathIntegrator(props))

if __name__ == "__main__":
    with dr.suspend_grad():
        scene: mi.Scene = mi.load_dict(mi.cornell_box())

        integrator: PathIntegrator = mi.load_dict(
            {
                "type": "path_test",
            }
        )

        sensor: mi.Sensor = scene.sensors()[0]
        size = sensor.film().crop_size()

        for i in range(10):
            img = integrator.render(scene, sensor, seed=i)
            mi.util.write_bitmap(f"out/{i}.jpg", img)
