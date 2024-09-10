from __future__ import annotations  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr


def concat_gather(arrays: list):
    final_width = 0
    for array in arrays:
        final_width += dr.width(array)

    index = dr.arange(mi.UInt, final_width)
    final_array = None

    for i in range(len(arrays)):
        array = arrays[i]
        gathered_array = dr.gather(
            type(array), array, index, index < dr.width(array)
        )  # relies on wrapping behaviour of UInt
        if final_array is None:
            final_array = gathered_array
        else:
            final_array = dr.select(
                index < dr.width(array), gathered_array, final_array
            )

        index = index - dr.width(array)

    return final_array


def concat_scatter(arrays):
    final_width = 0
    for array in arrays:
        final_width += dr.width(array)

    dst = dr.zeros(type(arrays[0]), shape=final_width)
    count = 0
    for array in arrays:
        n = dr.shape(array)[-1]
        i = dr.arange(mi.UInt32, count, count + n)
        dr.scatter(dst, array, i, i < final_width)

        count += n
    return dst


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

    sampler1: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler1.seed(0, 126)
    sampler2: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler2.seed(1, 2)

    a = sampler1.next_1d()
    b = sampler2.next_1d()

    result = concat([a, b])

    result2 = concat_scatter([a, b])

    assert dr.all(result == result2)
