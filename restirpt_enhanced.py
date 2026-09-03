import mitsuba as mi
import drjit as dr


from tqdm import tqdm
import os
import math
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
    fp: mi.Float
    valid: mi.Bool


@dataclass
class RestirReservoir:
    z: RestirSample
    w: mi.Float
    L: mi.Color3f
    W: mi.Float
    M: mi.Float

    def update(
        self,
        sampler: mi.Sampler,
        snew: RestirSample,
        wnew: mi.Color3f,
        cnew: mi.Float = 1.0,
        active: mi.Bool = True,
    ):
        active = mi.Bool(active)
        if dr.shape(active)[-1] == 1:
            dr.make_opaque(active)

        w = p_hat(wnew)

        self.L += dr.select(active, wnew, mi.Color3f(0))
        self.w += dr.select(active, w, 0)
        self.M += dr.select(active, cnew, 0)
        self.z: RestirSample = dr.select(
            active & (sampler.next_1d() * self.w < w), snew, self.z
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
        phat = p_hat(p)
        self.W = dr.select(phat * norm > 0, self.w / (phat * norm), 0)
        self.L = dr.select(norm > 0, self.L / norm, mi.Color3f(0))


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


class RestirPTEnhancedIntegrator(ADIntegrator):
    dist_threshold = 0.1
    angle_threshold = 25 * dr.pi / 180

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.bias_correction = props.get("bias_correction", False)
        self.use_jacobian = props.get("jacobian", True)
        self.bsdf_sampling = props.get("bsdf_sampling", True)
        self.max_M_temporal = props.get("max_M_temporal", 20)
        self.initial_search_radius = props.get("initial_search_radius", 30.0)
        self.spatial_rounds = props.get("spatial_rounds", 3)
        self.max_jacobian = props.get("max_jacobian", 3.0)
        self.footprint_threshold = props.get("footprint_threshold", 0.02)
        self.gaussian_neighbors = props.get("gaussian_neighbors", False)
        self.n = mi.UInt(0)
        self.film_size: None | mi.ScalarVector2u = None

    def prepare_frame(
        self,
        scene: mi.Scene,
        prev_sensor: int | mi.Sensor = 0,
        spp: int = 1,
    ):
        if isinstance(prev_sensor, int):
            prev_sensor = scene.sensors()[prev_sensor]

        film_size = prev_sensor.film().crop_size()

        if not hasattr(self, "prev_sensor"):
            self.film_size = film_size
            self.spp = spp

            n_samples = film_size.x * film_size.y * spp

            self.reservoir = dr.zeros(RestirReservoir, n_samples)
            self.prev_sample = dr.zeros(RestirSample, n_samples)
            self.search_radius = dr.full(
                mi.Float, self.initial_search_radius, n_samples
            )
            self.prev_sensor: mi.Sensor = mi.load_dict({"type": "perspective"})

        mi.traverse(self.prev_sensor).update(mi.traverse(prev_sensor))

    def to_idx(self, pos: mi.Point2u) -> mi.UInt:
        assert self.film_size is not None

        n_samples = self.film_size.x * self.film_size.y * self.spp
        sample_offset = dr.arange(mi.UInt, n_samples) % self.spp

        pos = dr.clip(mi.Point2u(pos), mi.Point2u(0), self.film_size - 1)
        return (pos.y * self.film_size.x + pos.x) * self.spp + sample_offset

    def sample_prev(
        self, sample: RestirSample
    ) -> tuple[RestirSample, RestirReservoir, mi.Bool]:
        """Lookup the reservoir and sample in the previous frame corresponding to this sample."""

        si_v: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si_v.p = sample.x_v
        ds, _ = self.prev_sensor.sample_direction(si_v, mi.Point2f(0.0))

        valid = ds.pdf > 0
        idx = self.to_idx(mi.Point2u(ds.uv))

        Sprev: RestirSample = dr.gather(RestirSample, self.prev_sample, idx, valid)
        Rprev: RestirReservoir = dr.gather(RestirReservoir, self.reservoir, idx, valid)

        return Sprev, Rprev, valid

    def similar(self, s1: RestirSample, s2: RestirSample) -> mi.Bool:
        dist = dr.norm(s1.x_v - s2.x_v)
        similar = dist < self.dist_threshold
        similar &= dr.dot(s1.n_v, s2.n_v) > dr.cos(self.angle_threshold)

        return similar

    def target(self, si: mi.SurfaceInteraction3f, S: RestirSample) -> mi.Color3f:
        d = S.x_s - si.p
        dist = dr.norm(d)
        active = S.valid & (dist > 0)
        f = si.bsdf().eval(mi.BSDFContext(), si, si.to_local(d / dist), active)
        return dr.select(active, f * S.L_o, mi.Color3f(0))

    def shift(self, receiver_pos: mi.Vector3f, S: RestirSample):
        valid = S.fp >= self.footprint_threshold / 100.0

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
        film_size = film.crop_size()

        if film.sample_border():
            raise Exception("sample_border=True is not supported by this integrator")

        if self.film_size is None:
            raise Exception("call prepare_frame(scene, sensor, spp) before rendering")

        if (self.film_size.x, self.film_size.y) != (film_size.x, film_size.y) or (
            self.spp != spp
        ):
            raise Exception(
                f"prepared for {self.film_size} at {self.spp} spp, but "
                f"rendering {film_size} at {spp} spp; call prepare_frame() again"
            )

        with dr.suspend_grad():
            sampler, spp = self.prepare(
                sensor=sensor, seed=seed, spp=spp, aovs=self.aov_names()
            )

            ray, weight, image_pos = self.sample_rays(scene, sensor, sampler)

            new_sample, si, Le_dir, specular, weight_v = self.sample_initial(
                scene, sampler, ray
            )
            dr.eval(new_sample)

            prev_sample, prev_reservoir, prev_valid = self.sample_prev(new_sample)

            reservoir = self.resample_temporal(
                scene,
                sampler,
                new_sample,
                si,
                specular,
                prev_sample,
                prev_reservoir,
                prev_valid,
            )
            dr.eval(reservoir)

            self.reservoir = self.resample_spatial(
                scene, sampler, new_sample, si, image_pos, reservoir
            )
            dr.eval(self.reservoir, self.search_radius)

            L = dr.select(specular, weight_v * new_sample.L_o, self.reservoir.L)
            L += Le_dir

            block = film.create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)

            ADIntegrator._splat_to_block(
                block,
                film,
                image_pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(si.is_valid(), mi.Float(1), mi.Float(0)),
                aovs=[],
                wavelengths=ray.wavelengths,
            )

            film.put_block(block)

            self.n += 1
            self.prev_sample = new_sample

            return film.develop()

    @dr.syntax
    def resample_spatial(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        new_sample: RestirSample,
        si: mi.SurfaceInteraction3f,
        image_pos: mi.Vector2f,
        reservoir: RestirReservoir,
    ) -> RestirReservoir:

        new_reservoir: RestirReservoir = dr.zeros(RestirReservoir)
        Q = dr.alloc_local(Reuse, self.spatial_rounds, value=dr.zeros(Reuse))

        new_reservoir.merge(
            sampler,
            reservoir,
            self.target(si, reservoir.z),
            mis=reservoir.M,
        )
        Z = mi.Float(reservoir.M)

        s = mi.UInt(0)
        while dr.hint(s < self.spatial_rounds, max_iterations=self.spatial_rounds):
            reuse = mi.Bool(True)

            if self.gaussian_neighbors:
                sigma = self.search_radius * math.sqrt(8.0 / (9.0 * math.pi))
                offset = dr.clip(
                    mi.warp.square_to_std_normal(sampler.next_2d()) * sigma,
                    -3.0 * sigma,
                    3.0 * sigma,
                )
            else:
                offset = (
                    mi.warp.square_to_uniform_disk(sampler.next_2d())
                    * self.search_radius
                )
            p = mi.Point2u(
                dr.clip(
                    mi.Point2i(image_pos) + mi.Point2i(offset),
                    mi.Point2i(0),
                    mi.Point2i(self.film_size) - 1,
                )
            )

            pi = mi.Point2u(
                dr.clip(
                    mi.Point2i(image_pos), mi.Point2i(0), mi.Point2i(self.film_size) - 1
                )
            )
            reuse &= (p.x != pi.x) | (p.y != pi.y)

            # Gather neighboring reservoir and sample
            neighbor_sample: RestirSample = dr.gather(
                RestirSample, new_sample, self.to_idx(p)
            )
            reuse &= self.similar(neighbor_sample, new_sample)
            neighbor_reservoir: RestirReservoir = dr.gather(
                RestirReservoir, reservoir, self.to_idx(p), reuse
            )

            jac, valid_shift = self.shift(new_sample.x_v, neighbor_reservoir.z)
            reuse &= valid_shift

            si_v: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
            si_v.p = new_sample.x_v
            si_v.n = new_sample.n_v
            shadowed = scene.ray_test(
                si_v.spawn_ray_to(neighbor_reservoir.z.x_s), reuse
            )

            phat = dr.select(
                reuse & ~shadowed, self.target(si, neighbor_reservoir.z), mi.Color3f(0)
            )

            new_reservoir.merge(
                sampler,
                neighbor_reservoir,
                phat,
                jacobian=jac,
                mis=neighbor_reservoir.M,
                active=reuse,
            )

            Q[s] = Reuse(
                mi.Float(neighbor_reservoir.M),
                mi.Vector3f(neighbor_reservoir.z.x_v),
                mi.Bool(reuse),
            )

            s += 1

        phat = self.target(si, new_reservoir.z)
        if self.bias_correction:
            i = mi.UInt(0)
            while dr.hint(i < self.spatial_rounds, max_iterations=self.spatial_rounds):
                r = Q[i]
                reuse = mi.Bool(r.active)

                si_s: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
                si_s.p = new_reservoir.z.x_s
                si_s.n = new_reservoir.z.n_s
                reuse &= ~scene.ray_test(si_s.spawn_ray_to(r.p), reuse)

                Z += dr.select(reuse, r.M, 0)

                i += 1

            new_reservoir.finalize(phat, Z)
        else:
            new_reservoir.finalize(phat, new_reservoir.M)

        new_reservoir.z.x_v = mi.Vector3f(new_sample.x_v)
        new_reservoir.z.n_v = mi.Vector3f(new_sample.n_v)

        return new_reservoir

    def resample_temporal(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        new_sample: RestirSample,
        si: mi.SurfaceInteraction3f,
        specular: mi.Bool,
        prev_sample: RestirSample,
        prev_reservoir: RestirReservoir,
        prev_valid: mi.Bool,
    ) -> RestirReservoir:
        valid = prev_valid & self.similar(new_sample, prev_sample)

        prev_reservoir = dr.select(valid, prev_reservoir, dr.zeros(RestirReservoir))
        if self.max_M_temporal is not None:
            prev_reservoir.M = dr.minimum(prev_reservoir.M, self.max_M_temporal)

        new_reservoir: RestirReservoir = dr.zeros(RestirReservoir)

        reuse = ~specular

        phat = self.target(si, new_sample)
        new_reservoir.update(
            sampler,
            new_sample,
            dr.select(new_sample.p_q > 0, phat / new_sample.p_q, mi.Color3f(0)),
            active=reuse,
        )

        jac, valid_shift = self.shift(new_sample.x_v, prev_reservoir.z)
        reuse &= valid_shift

        si_s: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        si_s.p = new_sample.x_v
        si_s.n = new_sample.n_v
        shadowed = scene.ray_test(si_s.spawn_ray_to(prev_reservoir.z.x_s), reuse)

        new_reservoir.merge(
            sampler,
            prev_reservoir,
            dr.select(
                reuse & ~shadowed, self.target(si, prev_reservoir.z), mi.Color3f(0)
            ),
            jacobian=jac,
            mis=prev_reservoir.M,
            active=reuse,
        )

        new_reservoir.finalize(self.target(si, new_reservoir.z), new_reservoir.M)

        new_reservoir.z.x_v = mi.Vector3f(new_sample.x_v)
        new_reservoir.z.n_v = mi.Vector3f(new_sample.n_v)

        return new_reservoir

    def sample_initial(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f):
        """
        Generate a new ReSTIR sample.
        """
        sample: RestirSample = dr.zeros(RestirSample, dr.width(ray.o))

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf(ray)

        ds = mi.DirectionSample3f(scene, si, dr.zeros(mi.SurfaceInteraction3f))
        Le_dir = ds.emitter.eval(si)

        sample.x_v = mi.Vector3f(si.p)
        sample.n_v = mi.Vector3f(si.n)

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

        sample.p_q = pdf

        ray = si.spawn_ray(si.to_world(wo))

        sample.L_o, pdf_s = self.sample_ray(scene, sampler, ray)

        si_s: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        sample.x_s = mi.Vector3f(si_s.p)
        sample.n_s = mi.Vector3f(si_s.n)
        sample.valid = si.is_valid() & si_s.is_valid() & (pdf > 0)

        v = sample.x_s - sample.x_v
        dist2 = dr.squared_norm(v)
        wo_w = v * dr.rsqrt(dist2)

        g_fwd = dr.abs(dr.dot(wo_w, sample.n_s)) / dist2
        g_rev = dr.abs(dr.dot(wo_w, sample.n_v)) / dist2

        fp_max = mi.Float(1e12)
        fp_fwd = dr.select(pdf * g_fwd > 0, dr.rcp(pdf * g_fwd), fp_max)
        fp_rev = dr.select(pdf_s * g_rev > 0, dr.rcp(pdf_s * g_rev), fp_max)

        cos_pri = dr.abs(dr.dot(sample.n_v, si.to_world(si.wi)))
        fp_pri = 4.0 * dr.pi * dr.square(si.t)

        sample.fp = dr.select(
            fp_pri > 0, dr.minimum(fp_fwd, fp_rev) * cos_pri / fp_pri, 0.0
        )

        return sample, si, Le_dir, specular, bsdf_weight

    @dr.syntax
    def sample_ray(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        active: bool = True,
    ) -> tuple[mi.Color3f, mi.Float]:
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
        pdf_s = mi.Float(0.0)

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

            pdf_s = dr.select(depth == 0, bsdf_sample.pdf, pdf_s)

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

        return dr.select(valid_ray, result, 0.0), pdf_s


mi.register_integrator(
    "restirpt_enhanced", lambda props: RestirPTEnhancedIntegrator(props)
)

if __name__ == "__main__":
    OUT = "out/restirpt-enhanced"
    RES = 1024
    FRAMES = 200
    os.makedirs(OUT, exist_ok=True)

    def camera(t: float) -> mi.ScalarTransform4f:
        angle = 0.25 * math.sin(2.0 * math.pi * t)
        return mi.ScalarTransform4f().look_at(
            origin=[3.9 * math.sin(angle), 0.0, 3.9 * math.cos(angle)],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
        )

    with dr.suspend_grad():
        scene = mi.cornell_box()
        del scene["small-box"]
        scene["glass-sphere"] = {
            "type": "sphere",
            "center": [0.335, -0.7, 0.38],
            "radius": 0.3,
            "bsdf": {"type": "dielectric", "int_ior": "bk7", "ext_ior": "air"},
        }
        scene: mi.Scene = mi.load_dict(scene)

        sensor: mi.Sensor = mi.load_dict(
            {
                "type": "perspective",
                "fov": 39.3077,
                "to_world": camera(0.0),
                "film": {
                    "type": "hdrfilm",
                    "width": RES,
                    "height": RES,
                    "rfilter": {"type": "box"},
                },
                "sampler": {"type": "independent"},
            }
        )

        print("Rendering Reference Image:")
        ref = mi.render(scene, sensor=sensor, spp=256)
        mi.util.write_bitmap(f"{OUT}/ref.exr", ref, write_async=False)

        integrator: RestirPTEnhancedIntegrator = mi.load_dict(
            {
                "type": "restirpt_enhanced",
                "jacobian": True,
                "bias_correction": False,
                "bsdf_sampling": True,
                "max_M_temporal": 20,
                "initial_search_radius": 30,
                "spatial_rounds": 3,
            }
        )

        render = dr.freeze(mi.render)

        for i in tqdm(range(FRAMES)):
            integrator.prepare_frame(scene, sensor, spp=1)

            params = mi.traverse(sensor)
            params["to_world"] = camera(i / FRAMES)
            params.update()

            img = render(
                scene, sensor=sensor, integrator=integrator, seed=mi.UInt32(i), spp=1
            )

            mi.util.write_bitmap(f"{OUT}/{i}.exr", img, write_async=False)
