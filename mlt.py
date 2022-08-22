

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")
from pathrecord import Path, drjitstruct  # noqa


@drjitstruct
class PathVert:
    wo: mi.Vector3f
    f: mi.Spectrum

    def __init__(self, wo=mi.Vector3f(), f=mi.Spectrum()):
        self.wo = wo
        self.f = f


class MltSampler:
    pass


class Simple(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get("max_depth")
        self.rr_depth = props.get("rr_depth")

    def render(self: mi.Integrator, scene: mi.Scene, sensor: mi.Sensor, seed: int = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> dr.scalar.TensorXf:
        film = sensor.film()
        sampler = sensor.sampler()

        spp = sampler.sample_count()
        self.spp = spp

        film_size = film.crop_size()
        n_chanels = film.prepare(self.aov_names())
        self.n_chanels = n_chanels

        wavefront_size = film_size.x * film_size.y

        # sampler.set_samples_per_wavefront()
        sampler.seed(0, wavefront_size)

        block: mi.ImageBlock = film.create_block()
        block.set_offset(film.crop_offset())

        idx = dr.arange(mi.UInt32, wavefront_size)

        pos = mi.Vector2f()
        pos.y = idx // film_size[0]
        pos.x = idx % film_size[0]

        pos += film.crop_offset()

        aovs = [mi.Float(0)] * n_chanels
        path_prev = Path(wavefront_size, self.max_depth, dtype=PathVert)

        print(spp)

        for i in range(spp):
            #X_new = dr.erfinv(sampler.next_2d())-X
            self.render_sample(scene, sensor, sampler,
                               block, aovs, pos, path_prev, idx)
            # Trigger kernel launch
            sampler.advance()
            sampler.schedule_state()
            dr.eval(path_prev.vertices)
            dr.eval(block.tensor())

        film.put_block(block)

        result = film.develop()
        dr.schedule(result)
        dr.eval()
        return result

    def render_sample(self, scene: mi.Scene, sensor: mi.Sensor, sampler: mi.Sampler, block: mi.ImageBlock, aovs, pos: mi.Vector2f, path_prev: Path, idx: mi.UInt32, active=True):
        film = sensor.film()
        scale = 1. / mi.Vector2f(film.crop_size())
        offset = - mi.Vector2f(film.crop_offset())
        sample_pos = pos + offset + sampler.next_2d()
        time = 1.
        s1, s3 = sampler.next_1d(), sampler.next_2d()
        ray, ray_weight = sensor.sample_ray(time, s1, sample_pos * scale, s3)
        medium = sensor.medium()

        active = mi.Bool(True)
        (spec, mask, aov) = self.sample(
            scene, sampler, ray, path_prev, idx, medium, active)

        spec = ray_weight * spec

        rgb = mi.Color3f()

        if mi.is_spectral:
            rgb = mi.spectrum_list_to_srgb(spec, ray.wavelengths, active)
        elif mi.is_monochromatic:
            rgb = spec.x
        else:
            rgb = spec

        # Debug:
        aovs[0] = rgb.x
        aovs[1] = rgb.y
        aovs[2] = rgb.z
        aovs[3] = 1.

        block.put(sample_pos, aovs)

    def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray_: mi.RayDifferential3f, path_prev: Path, idx: mi.UInt32, medium: mi.Medium = None, active: mi.Bool = True):
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(ray_)
        depth = mi.UInt32(0)
        f = mi.Spectrum(1.)
        L = mi.Spectrum(0.)
        active = mi.Bool(active)

        prev_si = dr.zeros(mi.SurfaceInteraction3f)

        loop = mi.Loop(name="Path Tracing", state=lambda: (
            sampler, ray, depth, f, L, active, prev_si))

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            """
            Xg = mi.Vector2f(
                dr.gather(mi.Float, X, index=(idx * self.spp * 2 + depth * 2)),
                dr.gather(mi.Float, X, index=(
                    idx * self.spp * 2 + depth * 2 + 1))
            )
            X_new = dr.erfinv(sampler.next_2d())-Xg

            """

            vert_prev: PathVert = path_prev[depth]
            wo_new = dr.erfinv(mi.warp.square_to_uniform_sphere(
                sampler.next_2d()))-vert_prev.wo

            si: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0))

            bsdf: mi.BSDF = si.bsdf(ray)

            # Direct emission

            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            Le = f * ds.emitter.eval(si)

            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # BSDF Sampling
            bsdf_smaple, bsdf_val = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next)

            # Update loop variables
            path_prev[depth] = PathVert(bsdf_smaple.wo, bsdf_val)

            ray = si.spawn_ray(si.to_world(bsdf_smaple.wo))
            L = (L + Le)
            f *= bsdf_val

            prev_si = dr.detach(si, True)

            # Stopping criterion (russian roulettte)

            active_next &= dr.neq(dr.max(f), 0)

            rr_prop = dr.maximum(f.x, dr.maximum(f.y, f.z))
            rr_prop[depth < self.rr_depth] = 1.
            f *= dr.rcp(rr_prop)
            active_next &= (sampler.next_1d() < rr_prop)

            active = active_next
            depth += 1
        return (L, dr.neq(depth, 0), [])


mi.register_integrator("integrator", lambda props: Simple(props))

scene = mi.cornell_box()
scene['integrator']['type'] = 'integrator'
scene['integrator']['max_depth'] = 16
scene['integrator']['rr_depth'] = 2
scene['sensor']['sampler']['sample_count'] = 16
scene['sensor']['film']['width'] = 1024
scene['sensor']['film']['height'] = 1024
scene = mi.load_dict(scene)

img = mi.render(scene)

plt.imshow(img ** (1. / 2.2))
plt.axis("off")
plt.show()
