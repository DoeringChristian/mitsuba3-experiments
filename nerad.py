# %% [markdown]
# Neural Radiosity


# %% [markdown]
# ## Imports
# Note, we require the `coopvec-hashgrid` branch of Dr.Jit, to support
# hashgrid encodings. We also recommend to use the `scatter-reduce-f16x2`
# branch for drjit-core. It will significantly improve performance of the
# hashgrids.

# %%

import mitsuba as mi
import drjit as dr
import drjit.nn as nn
import tqdm
import matplotlib.pyplot as plt
import os

mi.set_variant("cuda_ad_rgb")
from drjit.auto.ad import TensorXf16, Float16, Float32, Bool, UInt32
from drjit.opt import Adam, GradScaler
from mitsuba.ad.integrators.common import mis_weight


# %% [markdown]
# We define a seed generator function, which allows us to globally generate new
# seeds without having to worry, about accidentally generating the same seed
# twice.

# %%
__seed = 0


def seed():
    global __seed
    tmp = __seed
    __seed += 1
    return tmp


# %% [markdown]
# ## Radiance Field
# Let's define our radiance field model. It consists of a hash grid encoding,
# to encode the position of our samples, and a sequential model. With the new
# coopvec feature, we can declare our network with Dr.Jit, instead of having to
# go trough PyTorch. The network and encoding can now also be evaluated in the
# same kernel, that generates the surface interactions.

# %%


class Field:
    def __init__(
        self,
        scene: mi.Scene,
        width: int = 64,
        n_hidden: int = 4,
        sh_order: int = 3,
        bias: bool = False,
    ) -> None:
        self.scene = scene
        self.sh_order = sh_order

        self.pos_enc = nn.PermutoEncoding(Float16, dimension=3)

        sequential = []
        sequential.append(nn.Cast(Float16))
        sequential.append(
            nn.Linear(
                self.pos_enc.out_features + (sh_order + 1) ** 2 + 6, width, bias=bias
            )
        )
        sequential.append(nn.LeakyReLU())

        for i in range(n_hidden):
            sequential.append(nn.Linear(width, width, bias=bias))
            sequential.append(nn.LeakyReLU())

        sequential.append(nn.Linear(width, 3, bias=bias))

        sequential = nn.Sequential(*sequential)

        sequential = sequential.alloc(TensorXf16)

        self.weights, self.sequential = nn.pack(sequential, "training")

    def __call__(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        def normalize_pos(p: mi.Point3f):
            bbox = self.scene.bbox()
            return (p - bbox.min) / (bbox.max - bbox.min)

        p_norm = normalize_pos(si.p)

        p_enc = self.pos_enc(p_norm)

        wi_enc = dr.sh_eval(si.wi, self.sh_order)

        features = nn.CoopVec(p_norm, p_enc, si.wi, wi_enc)
        result = mi.Color3f(*self.sequential(features))

        return result


# %% [markdown]
# ## Integrator
# Most of the neural radiosity algorithm is implemented in this integrator. The
# functions `sample_rhs` and `sample_lhs` are used to sample the right and left
# hand side contributions of the equation.

# %%


class Integrator(mi.SamplingIntegrator):
    def __init__(self, field: Field) -> None:
        super().__init__(mi.Properties())
        self.field = field

    @dr.syntax
    def next_smooth_si(
        self,
        scene: mi.Scene,
        si: mi.SurfaceInteraction3f,
        sampler: mi.Sampler,
    ):
        """
        This function tries to find the next smooth surface interaction, by tracing
        a path in the scene.
        """

        f = mi.Spectrum(1)

        bsdf_ctx = mi.BSDFContext()

        bsdf = si.bsdf()
        bs, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d()
        )

        active = mi.Bool(mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta))
        depth = mi.UInt32(0)
        max_depth = 10

        while active:

            f *= bsdf_weight

            ray = si.spawn_ray(si.to_world(bs.wo))
            si = scene.ray_intersect(ray)

            bsdf = si.bsdf(ray)
            bs, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d()
            )

            depth += 1
            active &= mi.Bool(mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta))
            active &= depth < max_depth

        return si, f

    def sample_lhs(
        self,
        scene: mi.Scene,
        si: mi.SurfaceInteraction3f,
    ) -> mi.Color3f:
        L = self.field(si)
        return L

    def sample_rhs(
        self,
        scene: mi.Scene,
        si: mi.SurfaceInteraction3f,
        active: bool | Bool = True,
    ) -> mi.Color3f:
        active = Bool(active)

        index = dr.repeat(dr.arange(mi.UInt32, batch_size), M)
        si = dr.gather(type(si), si, index)
        active = dr.gather(type(active), active, index)

        active &= si.is_valid()

        sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        sampler.seed(seed(), batch_size * M)

        L = mi.Spectrum(0)

        bsdf = si.bsdf()
        bsdf_ctx = mi.BSDFContext()

        # Sample Emitters
        ds, emitter_value = scene.sample_emitter_direction(si, sampler.next_2d())
        bsdf_value, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, si.to_local(ds.d))

        L += mis_weight(ds.pdf, bsdf_pdf) * bsdf_value * emitter_value

        # Sample BSDF

        bs, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d()
        )

        ray = si.spawn_ray(si.to_world(bs.wo))
        si2 = scene.ray_intersect(ray)

        active &= si2.is_valid()

        ds = mi.DirectionSample3f(scene, si=si2, ref=si)
        emitter_pdf = scene.pdf_emitter_direction(
            si, mi.DirectionSample3f(scene, si=si2, ref=si)
        )
        f = mis_weight(bs.pdf, emitter_pdf) * bsdf_weight

        si2, f2 = self.next_smooth_si(scene, si2, sampler)
        active &= si2.is_valid()
        f *= f2
        f *= dr.select(active, 1.0, 0)

        # Reorder threads to improve performance
        si2 = dr.reorder_threads(UInt32(active), 1, si2)
        L_field = self.field(si2)

        emitter_value = si2.emitter(scene).eval(si2)
        L += f * (emitter_value + L_field)

        L = dr.block_sum(L, M, mode="symbolic") / M

        return L

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium | None = None,
        active: bool = True,
    ) -> tuple[mi.Spectrum, bool, list[float]]:

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        si = scene.ray_intersect(ray, active)
        si, f = self.next_smooth_si(scene, si, sampler)

        f *= dr.select(si.is_valid(), 1.0, 0.0)
        L = self.sample_lhs(scene, si) * f
        L += si.emitter(scene).eval(si)

        return L, active, []


# %%
batch_size = 2**14
M = 32

# %% [markdown]
# ## Surface Sampling
# To train the neural field, we have to sample points and directions on the
# surface of the scene. This class represents a sampler, which enables us to do
# so. In its constructor it constructs a distribution over shapes, proportional
# to their surface areas. When sampling a point on the surface, we use this to
# select a shape and then sample a point on the surface of that shape, for
# which Mitsuba has a function already implemented. The direction is sampled
# either on the sphere or hemisphere depending on if the bsdf of the shape is
# two sided.

# %%


class IntersectionSampler:
    """
    This class can be used to sample points on the surface of the scene,
    incomming directions. It returns surface interactions.
    """

    def __init__(self, scene: mi.Scene) -> None:
        self.scene = scene

        shapes = scene.shapes()
        weights = [shape.surface_area()[0] for shape in shapes]
        weights = mi.Float(weights)
        weights /= dr.sum(weights)

        self.distribution = mi.DiscreteDistribution(weights)

    def sample(self, sampler: mi.Sampler) -> mi.SurfaceInteraction3f:
        shapes: mi.ShapePtr = self.scene.shapes_dr()

        index = self.distribution.sample(sampler.next_1d())
        shape: mi.ShapePtr = dr.gather(mi.ShapePtr, shapes, index)

        dir_sample = sampler.next_2d()
        is_two_sided = mi.has_flag(shape.bsdf().flags(), mi.BSDFFlags.BackSide)

        ps: mi.PositionSample3f = shape.sample_position(0, sampler.next_2d())
        si = mi.SurfaceInteraction3f(ps=ps, wavelengths=[])
        si.initialize_sh_frame()
        si.shape = shape
        si.wi = dr.select(
            is_two_sided,
            mi.warp.square_to_uniform_sphere(dir_sample),
            mi.warp.square_to_uniform_hemisphere(dir_sample),
        )

        return si


# %%

scene_dict = mi.cornell_box()
scene_dict.pop("small-box")
scene_dict.pop("large-box")
scene_dict["glass"] = {"type": "dielectric"}
scene_dict["ball"] = {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.scale([0.4, 0.4, 0.4]).translate([0.5, 0, 0.5]),
    "bsdf": {"type": "ref", "id": "glass"},
}

scene = mi.load_dict(scene_dict)
# scene = mi.load_dict(mi.cornell_box())

integrator = mi.load_dict({"type": "path"})
img_ref = integrator.render(scene, scene.sensors()[0], seed(), 1_000)
mi.util.write_bitmap("out/nerad/ref.exr", img_ref)


field = Field(scene)

# Optimize a single-precision copy of the parameters
opt = Adam(
    lr=1e-3,
    params={
        "pos_enc": Float32(field.pos_enc.data),
        "sequential": Float32(field.weights),
    },
)

# This is an adaptive mixed-precision (AMP) optimization, where a half
# precision computation runs within a larger single-precision program.
# Gradient scaling is required to make this numerically well-behaved.
scaler = GradScaler()

sampler_lhs: mi.Sampler = mi.load_dict({"type": "independent"})
isampler = IntersectionSampler(scene)

integrator = Integrator(field)
os.makedirs("out/nerad", exist_ok=True)

# %% [markdown]
# ## Training Step
# We implement the training step in a separate function. This allows us to use
# frozen functions.

# %%


def training_step(scene, field, isampler, integrator, scaler):
    # Sample left-hand-side interaction
    si_lhs = isampler.sample(sampler_lhs)

    L_lhs = integrator.sample_lhs(scene, si_lhs)
    L_rhs = integrator.sample_rhs(scene, si_lhs)

    loss = dr.mean(dr.square(L_lhs - dr.detach(L_rhs)), axis=None)

    dr.backward(scaler.scale(loss))
    scaler.step(opt)

    return loss


# %% [markdown]
# Now we can use the training_step function to train a neural radiance field.

# %%

n = 1_000
val_interval = 100
iterator = tqdm.tqdm(range(n))
for it in iterator:
    sampler_lhs.seed(seed(), batch_size)

    field.weights[:] = Float16(opt["sequential"])
    field.pos_enc.data[:] = Float16(opt["pos_enc"])

    with dr.profile_range("training step"):
        loss = training_step(scene, field, isampler, integrator, scaler)

    if (it + 1) % val_interval == 0:
        with dr.profile_range("validation step"):
            loss = loss.numpy().item()

            img = integrator.render(scene, scene.sensors()[0], 0, 16)
            mse = dr.mean(dr.square(img - img_ref), axis=None).numpy().item()
            mi.util.write_bitmap(f"out/nerad/{it}.exr", img)

            iterator.set_postfix({"loss": loss, "mse": mse})
