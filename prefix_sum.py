import mitsuba as mi
import drjit as dr
import math

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def prefix_sum(x: mi.Float) -> mi.Float:
    """
    Implementation of a paralell prefix-sum described in

    W. Daniel Hillis and Guy L. Steele. 1986. Data parallel algorithms. Commun. ACM 29, 12 (Dec. 1986), 1170–1183. https://doi.org/10.1145/7902.7903
    """
    x = type(x)(x)

    loop_record = dr.flag(dr.JitFlag.LoopRecord)
    dr.set_flag(dr.JitFlag.LoopRecord, False)

    n = dr.shape(x)[-1]
    i = 0

    loop = mi.Loop("prefix-sum", lambda: ())

    while loop(i <= math.floor(math.log2(n))):
        j = dr.arange(mi.UInt, 2**i, n)
        if dr.shape(j)[-1] == 0:
            break
        res = dr.gather(type(x), x, j) + dr.gather(type(x), x, j - 2**i)
        dr.scatter(x, res, j)

        i += 1

    dr.set_flag(dr.JitFlag.LoopRecord, loop_record)

    return x


if __name__ == "__main__":
    import numpy as np

    sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler.seed(0, 1_000_000)

    x = sampler.next_1d()
    x_np = x.numpy()

    prefix_sum(x)
    x_np = np.cumsum(x_np)

    x = x.numpy()
    print(f"{x=}")
    print(f"{x_np=}")
    assert np.any(x != x_np)
