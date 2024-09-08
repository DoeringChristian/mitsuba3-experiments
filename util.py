from __future__ import annotations  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr


def concat(arrays: list):
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


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

    result = concat([mi.Float(1, 2, 3), mi.Float(1, 2)])
    print(f"{result=}")
