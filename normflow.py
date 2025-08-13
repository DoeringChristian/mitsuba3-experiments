# %%
import drjit as dr
import drjit.nn as nn
import imageio.v3 as iio

from drjit.opt import Adam, GradScaler
from drjit.auto.ad import (
    Texture2f,
    TensorXf,
    TensorXf16,
    Float16,
    Float32,
    Array2f,
    Array3f,
)

# %%

ref = TensorXf(
    iio.imread(
        "https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/wave-128.png"
    )
    / 256
)
tex = Texture2f(dr.mean(ref, axis=-1)[:, :, None])


# %% [markdown]
# Normalizing flows can be used to both sample from a learned distribution, but
# also evaluate the probability density function for a given sample. This makes
# them very useful in computer graphics, where both properties are often
# required.
#
# A normalizing flow is represented by an invertible function $f_\theta$. To
# sample random variables $X$ from the learned distribution, we sample latent
# variables $Z$ from a normal gaussian distribution $Z \sim p_Z = N(0, 1)$, and
# apply the inverse flow $X = f^{-1}_\theta(Z)$.
#
# We parameterize the normalizing flows with coupling and permutation layers
# $f_{i;\theta}$, such that $X = f_{0;\theta} \circ f_{1;\theta} \circ \dots
# f_{D;\theta} (Z)$. To train the network, we maximize the log sum of the
# estimated probability of sampling the sample i.e. $max \sum \text{log}
# p_{X;\theta}(X_i)$. To compute this probability, we can sum over the log
# determinant of the layers, $p_{X;\theta}(X) = \text{log} \left\vert \text{det} {\partial z
# \over \partial x} \right\vert_{\theta} + \text{log} p_{Z}(Z)$.

# %%


def uniform_to_std_normal(x: dr.ArrayBase):
    y = dr.zeros_like(x)
    r = dr.sqrt(-2.0 * dr.log(1.0 - x))
    phi = 2.0 * dr.pi * y

    c = dr.cos(phi)
    return c * r


def uniform_to_std_normal_pdf(z: dr.ArrayBase):
    return dr.inv_two_pi * dr.exp(-0.5 * dr.square(z))


class FlowLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def inverse(self, z: nn.CoopVec) -> nn.CoopVec: ...
    def forward(self, x: nn.CoopVec) -> tuple[nn.CoopVec, Float16]: ...


class PermutationLayer(FlowLayer):
    def __init__(self) -> None:
        super().__init__()

    def inverse(self, z: nn.CoopVec) -> nn.CoopVec:
        z = list(z)
        z.reverse()
        x = nn.CoopVec(z)
        return x

    def forward(self, x: nn.CoopVec) -> tuple[nn.CoopVec, Float16]:
        x = list(x)
        x.reverse()
        z = nn.CoopVec(x)
        ldj = Float16(0)
        return z, ldj


class CouplingLayer(FlowLayer):

    def __init__(self, n_layers: int = 3, n_activations: int = 64) -> None:
        super().__init__()

        sequential = []
        for i in range(n_layers - 1):
            sequential.append(nn.Linear(n_activations, n_activations))
            sequential.append(nn.ReLU())

        sequential.append(nn.Linear(n_activations, n_activations))

        self.net = nn.Sequential(*sequential)

    def inverse(self, z: nn.CoopVec) -> nn.CoopVec:
        r"""
        This function represents the inverse evaluation of the coupling layer,
        i.e. $X = f^{-1}_\theta(Z)$.
        """
        z: list = list(z)
        d = len(z) // 2

        id, z2 = z[:d, d:]

        p = list(self.net(nn.CoopVec(id)))
        a, mu = p[:d], p[d:]

        x2 = (z2 - mu) * dr.exp(-a)

        x = nn.CoopVec(id, x2)
        return x

    def forward(self, x: nn.CoopVec) -> tuple[nn.CoopVec, Float16]:
        r"""
        This function evaluates the foward flow $Z = f_\theta(X)$, as well as
        the log jacobian determinant.
        """

        x = list(x)
        d = len(x) // 2

        id, x2 = x[:d], x[d:]

        p = list(self.net(nn.CoopVec(id)))
        a, mu = p[:d], p[d:]

        z2 = x2 * dr.exp(a) + mu
        z = nn.CoopVec(id, z2)
        ldj = dr.sum(a)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()

        layers: list[FlowLayer] = []
        for i in range(n_layers):
            layers.append(CouplingLayer())
            layers.append(PermutationLayer())

        self.layers = layers

    def sample_base_dist(self, sample: nn.CoopVec) -> nn.CoopVec:
        return nn.CoopVec(*[uniform_to_std_normal(x) for x in sample])

    def eval_base_dist(self, z: nn.CoopVec) -> dr.ArrayBase:
        return dr.prod([uniform_to_std_normal_pdf(z) for z in z])

    def log_p(self, x: nn.CoopVec) -> Float16:
        """
        This function calculates the log probability of sampling a given value
        `x`.
        """

        log_p = dr.zeros(x.dtype)

        for layer in self.layers:
            x, ldj = layer.forward(x)
            log_p += ldj

        z = x

        log_p += dr.log(self.eval_base_dist(z))
        return log_p

    def sample(self, sample: nn.CoopVec) -> nn.CoopVec:
        r"""
        Sample a function from the learned target distribution $X \sim
        p_{X;\theta}$, given a sample from the uniform distribution.
        """
        z = self.sample_base_dist(sample)

        for layer in reversed(self.layers):
            z = layer.inverse(z)

        return z
