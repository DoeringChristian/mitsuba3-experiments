from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr


def w2c(sensor: mi.ProjectiveCamera) -> mi.Transform4f:
    params = mi.traverse(sensor)

    C2S = mi.perspective_projection(
        params["film.size"],
        params["film.crop_size"],
        params["film.crop_offset"],
        params["x_fov"],
        params["near_clip"],
        params["far_clip"],
    )
    W2C = params["to_world"]
    return C2S @ W2C


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    scene = mi.load_dict(mi.cornell_box())  # type:  mi.Scene

    params = mi.traverse(scene.sensors()[0])

    C2S = mi.perspective_projection(
        params["film.size"],
        params["film.crop_size"],
        params["film.crop_offset"],
        params["x_fov"],
        params["near_clip"],
        params["far_clip"],
    )
    W2C = params["to_world"]
    p = mi.Point3f(0.0, 0.0, -100.0)
    s = C2S @ (W2C @ p)
    print(f"{s=}")
