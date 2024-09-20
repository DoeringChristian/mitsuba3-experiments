import drjit as dr
import mitsuba as mi
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def interval_to_exp(sample: mi.Float):
    return -dr.log(1 - sample + 1e-8)


def normal_pdf(x: mi.Float, mu: mi.Float, sigma: mi.Float):
    return (
        (dr.inv_sqrt_two_pi)
        * (1.0 / sigma)
        * dr.exp(-dr.square((x - mu)) / (2 * dr.square(sigma)))
    )


class MetropolisSampler(mi.Sampler):
    """
    Implementation of the Metropolis sampler, that works with python loops, but not with mitsuba loops.
    """

    def __init__(self, sigma=0.1, p_large=0.1) -> None:
        super().__init__(mi.Properties())
        self.sigma = dr.opaque(mi.Float, sigma)
        self.p_large = dr.opaque(mi.Float, p_large)

        # State variables
        self.independent: mi.Sampler = mi.load_dict({"type": "independent"})
        self.proposed = []
        self.i = 0
        self.f = mi.Float(0)
        self.samples = None

        self.wavefront_size = 0

    def seed(self, seed=0, wavefront_size=1024):
        self.independent.seed(seed, wavefront_size)
        self.wavefront_size = wavefront_size

    def initial_1d(self, active: mi.Bool) -> mi.Float:
        return self.independent.next_1d(active)

    def next_1d(self, active: mi.Bool = True) -> mi.Float:
        if len(self.proposed) > self.i:
            result = self.proposed[self.i]
        else:
            result = self.initial_1d(active)
            self.proposed.append(result)
        self.i += 1
        return result

    def next_2d(self, active: mi.Bool = True) -> mi.Point2f:
        return mi.Point2f(self.next_1d(active), self.next_1d(active))

    def sample_proposal(self, x: mi.Float) -> mi.Float:
        y = x + mi.warp.square_to_std_normal(self.independent.next_2d()).x * self.sigma
        y = y - dr.floor(y)

        large = self.independent.next_1d() < self.p_large
        y = dr.select(large, self.independent.next_1d(), y)

        return y

    def pdf_proposal(self, x: mi.Float, y: mi.Float) -> mi.Float:
        return normal_pdf(x, y, self.sigma)

    def advance(self, f: mi.Float):
        acceptance = dr.minimum(1, f / self.f)
        accept = self.independent.next_1d() <= acceptance

        if self.samples:
            self.samples = [
                dr.select(accept, proposed, sample)
                for sample, proposed in zip(self.samples, self.proposed)
            ]
        else:
            self.samples = [mi.Float(sample) for sample in self.proposed]

        # Update with new proposal
        self.proposed = [self.sample_proposal(x) for x in self.samples]
        self.f = mi.Float(f)
        self.i = 0

    def schedule_state(self):
        self.independent.schedule_state()
        dr.schedule(self.samples)
        dr.schedule(self.proposed)
        dr.schedule(self.f)

    def set_sample_count(self, spp: int):
        self.spp = spp

    def sample_count(self) -> int:
        return self.spp

    def set_samples_per_wavefront(self, spp_per_pass: int):
        self.spp_per_pass = spp_per_pass


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def target(x):

    def f(x):
        return gaussian(x, 0.2, 0.01) + gaussian(x, 0.7, 0.1)

    between_0_1 = np.logical_and(0.0 < x, x < 1.0)
    outside_05_06 = np.logical_or(x < 0.5, 0.6 < x)

    range = np.logical_and(between_0_1, outside_05_06)

    target = np.select([range], [f(x)], 0)

    return target


def Dkl(p, q):
    return np.nanmean(np.where(p > 0, p * np.log(p / q), 0))


def KL(P: np.ndarray, Q: np.ndarray) -> float:
    epsilon = 0.00001

    P = P + epsilon
    Q = Q + epsilon

    divergence = np.mean(P * np.log(P / Q))
    return divergence


def test(name: str, iterations, n, bins, sampler):
    x_ref = np.linspace(0, 1, 1000)
    y_ref = target(x_ref)
    y_ref = y_ref / np.mean(y_ref)

    kl_distances = []

    sampler.seed(0, n)
    iterator = tqdm.tqdm(range(iterations))
    for i in iterator:
        dr.kernel_history_clear()

        x = sampler.next_1d().numpy()

        f = target(x)

        sampler.advance(mi.Float(f))
        sampler.schedule_state()
        dr.eval()

        if i % 10 == 0:
            kde = stats.gaussian_kde(x)
            plt.clf()
            plt.hist(x, bins=bins, density=True, label="Metropolis Histogram")
            plt.plot(x_ref, y_ref, label="Ref")
            plt.plot(x_ref, kde(x_ref), label="Metropolis KDE")
            plt.legend()
            os.makedirs(f"out/{name}", exist_ok=True)
            plt.savefig(f"out/{name}/{i}.svg")

            target_pdf = target(np.linspace(0, 1, bins))
            target_pdf = target_pdf / np.mean(target_pdf)
            sample_pdf = np.histogram(x, bins, density=True)[0]
            dkl = KL(sample_pdf, target_pdf)
            iterator.set_postfix({"dkl": dkl})

            kl_distances.append((i, dkl))

    it, kl_distances = zip(*kl_distances)

    plt.clf()
    plt.plot(it, kl_distances)
    plt.xlabel("iteration")
    plt.ylabel("$D_{KL}$")
    plt.yscale("log")
    plt.savefig(f"out/{name}/dkl.svg")

    return it, kl_distances


if __name__ == "__main__":
    iterations = 1_000
    n = 16384
    bins = 128
    dr.set_flag(dr.JitFlag.KernelHistory, True)

    sampler = MetropolisSampler(0.001, 0.01)
    metropolis = test("metropolis", iterations, n, bins, sampler)

    # print(f"{metropolis=}")
    # print(f"{jump_restore=}")

    plt.clf()
    plt.plot(metropolis[0], metropolis[1], label="Metropolis")
    plt.xlabel("iteration")
    plt.ylabel("$D_{KL}$")
    plt.yscale("log")
    plt.legend()
    plt.savefig("out/dkl.svg")
