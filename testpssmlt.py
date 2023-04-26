import gc
from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")
# dr.set_log_level(dr.LogLevel.Debug)
# mi.set_log_level(mi.LogLevel.Debug)
import pssmltpath
import pssmltsimple
import simple
import pssmlt


if __name__ == "__main__":
    scene = mi.load_file("data/caustics/scene.xml")

    img = None
    with dr.suspend_grad():
        integrator = mi.load_dict(
            {
                "type": "ptracer",
            }
        )  # type: ignore
        ref_pt = mi.render(scene, integrator=integrator, spp=128)
        mi.util.write_bitmap("out/ref_pt.png", ref_pt)

        integrator = mi.load_dict(
            {
                "type": "path",
            }
        )  # type: ignore
        ref_path = mi.render(scene, integrator=integrator, spp=128)
        mi.util.write_bitmap("out/ref_path.png", ref_path)

        integrator: pssmltsimple.Pssmlt = mi.load_dict(
            {
                "type": "pssmlt",
                "max_depth": 8,
                "rr_depth": 2,
            }
        )  # type: ignore
        img = integrator.render(scene, scene.sensors()[0], seed=0, spp=4)
        mi.util.write_bitmap("out/img.png", img)

        mi.util.write_bitmap("out/dif-pt.png", img - ref_pt)
        mi.util.write_bitmap("out/dif-pt-1.png", ref_pt - img)
