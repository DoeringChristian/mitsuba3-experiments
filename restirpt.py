import mitsuba as mi
import drjit as dr


from tqdm import tqdm
import os
from dataclasses import dataclass

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

from mitsuba.ad.integrators.common import ADIntegrator, mis_weight


@dataclass
class RestirSample:
    """
    A ReSTIRSample, representing
    """

    x_v: mi.Vector3f
    n_v: mi.Vector3f
    x_s: mi.Vector3f
    n_s: mi.Vector3f

    L_o: mi.Color3f
    p_q: mi.Float
    valid: mi.Bool


@dataclass
class RestirReservoir:
    z: RestirSample
    w: mi.Float
    W: mi.Float
    M: mi.Float

    def update(
        self,
        sampler: mi.Sampler,
        snew: RestirSample,
        wnew: mi.Float,
        cnew: mi.Float = 1.0,
        active: mi.Bool = True,
    ):
        active = mi.Bool(active)
        if dr.shape(active)[-1] == 1:
            dr.make_opaque(active)

        self.w += dr.select(active, wnew, 0)
        self.M += dr.select(active, cnew, 0)
        self.z: RestirSample = dr.select(
            active & (sampler.next_1d() * self.w < wnew), snew, self.z
        )

    def merge(
        self,
        sampler: mi.Sampler,
        r: "RestirReservoir",
        p,
        jacobian: mi.Float = 1.0,
        mis: mi.Float = 1.0,
        active: mi.Bool = True,
    ):
        self.update(sampler, r.z, mis * p * r.W * jacobian, r.M, active)

    def finalize(self, p, norm: mi.Float = 1.0):
        self.W = dr.select(p * norm > 0, self.w / (p * norm), 0)


def J(receiver_pos: mi.Vector3f, S: RestirSample) -> mi.Float:
    v_new = receiver_pos - S.x_s
    d_new = dr.norm(v_new)
    cos_new = dr.abs(dr.dot(v_new / d_new, S.n_s))

    v_old = S.x_v - S.x_s
    d_old = dr.norm(v_old)
    cos_old = dr.abs(dr.dot(v_old / d_old, S.n_s))

    div = cos_old * dr.square(d_new)
    return dr.select(div > 0, cos_new * dr.square(d_old) / div, 0)


def p_hat(f) -> mi.Float:
    return mi.luminance(f)


@dataclass
class Reuse:
    M: mi.Float
    p: mi.Vector3f
    active: mi.Bool


class RestirPTIntegrator(ADIntegrator):
    dist_threshold = 0.1
    angle_threshold = 25 * dr.pi / 180

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.bias_correction = props.get("bias_correction", True)
        self.use_jacobian = props.get("jacobian", True)
        self.bsdf_sampling = props.get("bsdf_sampling", True)
        self.max_M_temporal = props.get("max_M_temporal", 20)
        self.max_M_spatial = props.get("max_M_spatial", 20)
        self.initial_search_radius = props.get("initial_search_radius", 10.0)
        self.minimal_search_radius = props.get("minimal_search_radius", 3.0)
        self.spatial_rounds = props.get("spatial_rounds", 9)
        self.max_jacobian = props.get("max_jacobian", 3.0)
        self.min_reconnect_distance = props.get("min_reconnect_distance", 0.05)
        self.jitter = props.get("jitter", False)
        self.n = mi.UInt(0)
        self.film_size: None | mi.ScalarVector2u = None

    def init(self, scene: mi.Scene, sensor: int | mi.Sensor = 0, spp: int = 1):
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        self.film_size = film.crop_size()
        self.spp = spp

        n_samples = self.film_size.x * self.film_size.y * spp

        self.reservoir = dr.zeros(RestirReservoir, n_samples)
        self.prev_sample = dr.zeros(RestirSample, n_samples)
        self.search_radius = dr.full(mi.Float, self.initial_search_radius, n_samples)

        self.prev_sensor: mi.Sensor = mi.load_dict({"type": "perspective"})
        mi.traverse(self.prev_sensor).update(mi.traverse(sensor))

    def to_idx(self, pos: mi.Point2u) -> mi.UInt:
        assert self.film_size is not None
        pos = dr.clip(mi.Point2u(pos), mi.Point2u(0), self.film_size - 1)
        return (pos.y * self.film_size.x + pos.x) * self.spp + self.sample_offset

    def similar(self, s1: RestirSample, s2: RestirSample) -> mi.Bool:
        dist = dr.norm(s1.x_v - s2.x_v)
        similar = dist < self.dist_threshold
        similar &= dr.dot(s1.n_v, s2.n_v) > dr.cos(self.angle_threshold)

        return similar

    def target(self, si: mi.SurfaceInteraction3f, S: RestirSample) -> mi.Float:
        d = S.x_s - si.p
        dist = dr.norm(d)
        active = S.valid & (dist > 0)
        f = si.bsdf().eval(mi.BSDFContext(), si, si.to_local(d / dist), active)
        return dr.select(active, p_hat(f * S.L_o), 0)

    def shift(self, receiver_pos: mi.Vector3f, S: RestirSample):
        d_min = self.min_reconnect_distance
        valid = (dr.norm(S.x_s - S.x_v) >= d_min) & (
            dr.norm(S.x_s - receiver_pos) >= d_min
        )

        if not self.use_jacobian:
            return mi.Float(1.0), valid

        jac = J(receiver_pos, S)
        valid &= (jac > 1.0 / self.max_jacobian) & (jac < self.max_jacobian)

        return jac, valid

    def render(
        self,
        scene: mi.Scene,
        sensor: int | mi.Sensor = 0,
        seed: int = 0,
        spp: int = 0,
        develop: bool = True,
        evaluate: bool = True,
    ) -> mi.TensorXf:
        if not develop:
            raise Exception(
                "develop=True must be specified when invoking this integrator"
            )

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        if film.sample_border():
            raise Exception("sample_border=True is not supported by this integrator")

        with dr.suspend_grad():
            sampler, spp = self.prepare(
                sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names()
            )

            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            if not self.jitter:
                scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
                offset = -mi.ScalarVector2f(film.crop_offset()) * scale
                pos = dr.floor(pos) + 0.5
                ray, weight = sensor.sample_ray_differential(
                    time=sensor.shutter_open(),
                    sample1=0.0,
                    sample2=dr.fma(pos, scale, offset),
                    sample3=mi.Point2f(0.5),
                )

            film_size = film.crop_size()
            if self.film_size is None:
                raise Exception("call init(scene, sensor, spp) before rendering")

            if (self.film_size.x, self.film_size.y) != (film_size.x, film_size.y) or (
                self.spp != spp
            ):
                raise Exception(
                    f"initialized for {self.film_size} at {self.spp} spp, but "
                    f"rendering {film_size} at {spp} spp; call init() again"
                )

            self.sample_offset = dr.arange(mi.UInt, dr.width(ray.o)) % spp
            self.pos = mi.Point2u(pos)

            sample, si, Le_dir, specular, weight_v = self.sample_initial(
                scene, sampler, ray
            )
            dr.eval(sample)

            temporal = self.temporal_resampling(scene, sampler, sample, si, specular)
            dr.eval(temporal)

            self.reservoir = self.spatial_resampling(
                scene, sampler, sample, si, temporal
            )
            dr.eval(self.reservoir, self.search_radius)

            L = self.render_final(sample, si, Le_dir, specular, weight_v)

            block = film.create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)

            ADIntegrator._splat_to_block(
                block,
                film,
                pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(si.is_valid(), mi.Float(1), mi.Float(0)),
                aovs=[],
                wavelengths=ray.wavelengths,
            )

            film.put_block(block)

            self.n += 1
            mi.traverse(self.prev_sensor).update(mi.traverse(sensor))
            self.prev_sample = sample

            return film.develop()

    def render_final(
        self,
        sample: RestirSample,
        si: mi.SurfaceInteraction3f,
        Le_dir: mi.Color3f,
        specular: mi.Bool,
        weight_v: mi.Color3f,
    ) -> mi.Color3f:
        R = self.reservoir
        S = R.z

        d = S.x_s - si.p
        dist = dr.norm(d)
        active = S.valid & (dist > 0)
        f = si.bsdf().eval(mi.BSDFContext(), si, si.to_local(d / dist), active)

        L = dr.select(specular, weight_v * sample.L_o, f * S.L_o * R.W)

        return L + Le_dir

    @dr.syntax
    def spatial_resampling(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        sample: RestirSample,
        si: mi.SurfaceInteraction3f,
        temporal: RestirReservoir,
    ) -> RestirReservoir:
        Rs = self.reservoir

        Rnew: RestirReservoir = dr.zeros(RestirReservoir)
        Q = dr.alloc_local(Reuse, self.spatial_rounds, value=dr.zeros(Reuse))

        pos = self.pos
        q: RestirSample = sample

        R = temporal
        Rnew.merge(sampler, R, self.target(si, R.z), mis=R.M)
        Z = mi.Float(R.M)

        max_iter = (
            dr.select(
                Rs.M < self.max_M_spatial / 2, mi.UInt(self.spatial_rounds), mi.UInt(3)
            )
            if self.max_M_spatial is not None
            else dr.full(mi.UInt, self.spatial_rounds, dr.width(pos.x))
        )

        any_reused = dr.full(mi.Bool, False, dr.width(pos.x))

        s = mi.UInt(0)
        while dr.hint(s < max_iter, max_iterations=self.spatial_rounds):
            active = mi.Bool(True)

            offset = (
                mi.warp.square_to_uniform_disk(sampler.next_2d()) * self.search_radius
            )
            p = mi.Point2u(
                dr.clip(
                    mi.Point2i(pos) + mi.Point2i(offset),
                    mi.Point2i(0),
                    mi.Point2i(self.film_size) - 1,
                )
            )

            qn: RestirSample = dr.gather(RestirSample, sample, self.to_idx(p))

            active &= self.similar(qn, q)

            Rn: RestirReservoir = dr.gather(
                RestirReservoir, temporal, self.to_idx(p), active
            )

            jac, shift_ok = self.shift(q.x_v, Rn.z)
            active &= shift_ok

            si_v: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
            si_v.p = q.x_v
            si_v.n = q.n_v
            shadowed = scene.ray_test(si_v.spawn_ray_to(Rn.z.x_s), active)

            phat = dr.select(active & ~shadowed, self.target(si, Rn.z), 0)

            Rnew.merge(
                sampler,
                Rn,
                phat,
                jacobian=jac,
                mis=Rn.M,
                active=active,
            )

            Q[s] = Reuse(mi.Float(Rn.M), mi.Vector3f(Rn.z.x_v), mi.Bool(active))

            any_reused |= active

            s += 1

        phat = self.target(si, Rnew.z)
        if self.bias_correction:
            i = mi.UInt(0)
            while dr.hint(i < self.spatial_rounds, max_iterations=self.spatial_rounds):
                r = Q[i]
                active = mi.Bool(r.active)

                si_s: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
                si_s.p = Rnew.z.x_s
                si_s.n = Rnew.z.n_s
                active &= ~scene.ray_test(si_s.spawn_ray_to(r.p), active)

                Z += dr.select(active, r.M, 0)

                i += 1

            Rnew.finalize(phat, Z)
        else:
            Rnew.finalize(phat, Rnew.M)

        Rnew.z.x_v = mi.Vector3f(q.x_v)
        Rnew.z.n_v = mi.Vector3f(q.n_v)

        self.search_radius = dr.maximum(
            dr.select(any_reused, self.search_radius, self.search_radius / 2),
            self.minimal_search_radius,
        )

        if self.max_M_spatial is not None:
            Rnew.M = dr.minimum(Rnew.M, self.max_M_spatial)

        return Rnew

    def temporal_resampling(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        sample: RestirSample,
        si: mi.SurfaceInteraction3f,
        specular: mi.Bool,
    ) -> RestirReservoir:
        si_v: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si_v.p = sample.x_v
        ds, _ = self.prev_sensor.sample_direction(si_v, mi.Point2f(0.0))

        valid = ds.pdf > 0

        Sprev: RestirSample = dr.gather(
            RestirSample, self.prev_sample, self.to_idx(mi.Point2u(ds.uv)), valid
        )

        valid &= self.similar(sample, Sprev)

        R = dr.select(valid, self.reservoir, dr.zeros(RestirReservoir))

        Rnew: RestirReservoir = dr.zeros(RestirReservoir)

        reuse = ~specular

        phat = self.target(si, sample)
        Rnew.update(
            sampler,
            sample,
            dr.select(sample.p_q > 0, phat / sample.p_q, 0),
            active=reuse,
        )

        jac, shift_ok = self.shift(sample.x_v, R.z)
        reuse &= shift_ok

        si_s: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si_s.p = sample.x_v
        si_s.n = sample.n_v
        shadowed = scene.ray_test(si_s.spawn_ray_to(R.z.x_s), reuse)

        Rnew.merge(
            sampler,
            R,
            dr.select(reuse & ~shadowed, self.target(si, R.z), 0),
            jacobian=jac,
            mis=R.M,
            active=reuse,
        )

        Rnew.finalize(self.target(si, Rnew.z), Rnew.M)

        Rnew.z.x_v = mi.Vector3f(sample.x_v)
        Rnew.z.n_v = mi.Vector3f(sample.n_v)

        if self.max_M_temporal is not None:
            Rnew.M = dr.minimum(Rnew.M, self.max_M_temporal)

        return Rnew

    def sample_initial(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f):
        S: RestirSample = dr.zeros(RestirSample, dr.width(ray.o))

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf(ray)

        ds = mi.DirectionSample3f(scene, si, dr.zeros(mi.SurfaceInteraction3f))
        Le_dir = ds.emitter.eval(si)

        S.x_v = mi.Vector3f(si.p)
        S.n_v = mi.Vector3f(si.n)

        bsdf_sample, bsdf_weight = bsdf.sample(
            mi.BSDFContext(), si, sampler.next_1d(), sampler.next_2d()
        )

        specular = ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        if self.bsdf_sampling:
            wo = bsdf_sample.wo
            pdf = bsdf_sample.pdf
        else:
            wo_u = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
            pdf_u = mi.warp.square_to_uniform_hemisphere_pdf(wo_u)

            wo = dr.select(specular, bsdf_sample.wo, wo_u)
            pdf = dr.select(specular, bsdf_sample.pdf, pdf_u)

        S.p_q = pdf

        ray = si.spawn_ray(si.to_world(wo))

        S.L_o = self.sample_ray(scene, sampler, ray)

        si_s: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        S.x_s = mi.Vector3f(si_s.p)
        S.n_s = mi.Vector3f(si_s.n)
        S.valid = si.is_valid() & si_s.is_valid() & (pdf > 0)

        return S, si, Le_dir, specular, bsdf_weight

    @dr.syntax
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

        while dr.hint(active, max_iterations=self.max_depth, label="Path Tracer"):
            si = scene.ray_intersect(ray)

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

            rr_prop = dr.minimum(throughput_max * dr.square(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prop

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = active_next & (~rr_active | rr_continue) & (throughput_max != 0.0)

        return dr.select(valid_ray, result, 0.0)


mi.register_integrator("restirpt", lambda props: RestirPTIntegrator(props))

if __name__ == "__main__":
    OUT = "out/restirpt"
    os.makedirs(OUT, exist_ok=True)

    with dr.suspend_grad():
        scene = mi.cornell_box()
        scene["sensor"]["film"]["width"] = 1024
        scene["sensor"]["film"]["height"] = 1024
        scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
        del scene["small-box"]
        scene["glass-sphere"] = {
            "type": "sphere",
            "center": [0.335, -0.7, 0.38],
            "radius": 0.3,
            "bsdf": {"type": "dielectric", "int_ior": "bk7", "ext_ior": "air"},
        }
        scene: mi.Scene = mi.load_dict(scene)

        print("Rendering Reference Image:")
        ref = mi.render(scene, spp=256)
        mi.util.write_bitmap(f"{OUT}/ref.exr", ref)

        integrator: RestirPTIntegrator = mi.load_dict(
            {
                "type": "restirpt",
                "jacobian": True,
                "bias_correction": True,
                "bsdf_sampling": True,
                "max_M_spatial": 20,
                "max_M_temporal": 20,
                "initial_search_radius": 10,
            }
        )

        integrator.init(scene, spp=1)

        render = dr.freeze(mi.render)

        print("ReSTIR PT:")
        for i in tqdm(range(200)):
            img = render(scene, integrator=integrator, seed=i, spp=1)

            mi.util.write_bitmap(f"{OUT}/{i}.exr", img)
