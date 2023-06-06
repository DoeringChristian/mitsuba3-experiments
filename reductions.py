from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def scatter_reduce_with(func, target, value, index, active=True):
    # n_value = dr.shape(value)[-1]
    # n_target = dr.shape(target)[-1]
    n_value = dr.width(value)
    n_target = dr.width(target)
    # print(f"{n_value=}")
    # print(f"{n_target=}")

    current_scatter = dr.zeros(mi.UInt, n_target)
    queued_values = dr.arange(mi.UInt, n_value)

    while len(queued_values) > 0:
        """
        First we scatter into the `current_scatter` array.
        For every double index, a random element is selected
        """

        target_idx = dr.gather(mi.UInt, index, queued_values)
        lane_idx = dr.gather(mi.UInt, dr.arange(mi.UInt, n_value), queued_values)
        dr.scatter(
            current_scatter,
            lane_idx,
            dr.gather(mi.UInt, index, queued_values),
        )

        """
        We now get the selected values for scattering in this loop iteration
        """
        current = dr.eq(dr.gather(mi.UInt, current_scatter, target_idx), lane_idx)

        current_idx = dr.gather(mi.UInt, queued_values, dr.compress(current))

        queued_values = dr.gather(mi.UInt, queued_values, dr.compress(~current))

        target_idx = dr.gather(mi.UInt, index, current_idx)

        a = dr.gather(type(target), target, target_idx)
        b = dr.gather(type(value), value, current_idx)
        """
        After gathering the target and values of the current lanes we compute the result
        """
        res = func(a, b)
        dr.scatter(target, res, target_idx)


if __name__ == "__main__":
    target = dr.zeros(mi.Float, 10)
    index = dr.arange(mi.UInt, 25) % 10
    value = dr.ones(mi.Float, 25)

    scatter_reduce_with(lambda a, b: a + b, target, value, index)
    print(f"{target=}")
