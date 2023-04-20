from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")
import pssmltpath
import pssmltsimple


scene = mi.cornell_box()
scene["sensor"]["film"]["width"] = 1024
scene["sensor"]["film"]["height"] = 1024
scene["sphere"] = {
    "type": "sphere",
    "to_world": mi.ScalarTransform4f.translate([0.335, -0.7, 0.38]).scale(0.3),
    "bsdf": {"type": "dielectric"},
}
scene["blocking"] = {
    "type": "cube",
    "to_world": mi.ScalarTransform4f.translate([0.0, 0.4, 0.0]).scale(0.3),
}
del scene["small-box"]
print(f"{scene=}")
scene = mi.load_dict(scene)


def render_pssmlt(n=100, seed=0):
    integrator: pssmltsimple.Pssmlt = mi.load_dict(
        {
            "type": "pssmlt_simple",
            "max_depth": 8,
            "rr_depth": 2,
        }
    )
    img = None
    mlt_depth = 50
    j = 1
    for i in range(n):
        print(f"{i=}")
        # nimg = mi.render(scene, integrator=integrator, seed=i + seed * n, spp=1)
        nimg = integrator.render(scene, scene.sensors()[0], seed=i + seed * n, spp=1)
        if i < mlt_depth:
            img = nimg
        else:
            img = img * mi.Float((j - 1) / j) + nimg / mi.Float(j)
            j += 1
        # mi.util.write_bitmap(f"out/i{i}.png", img, write_async=False)
    return img


img = None
j = 1
with dr.suspend_grad():
    for i in range(10):
        nimg = render_pssmlt(seed=i)

        if img is None:
            img = nimg
        else:
            img = img * mi.Float((j - 1) / j) + nimg / mi.Float(j)
            j += 1
        mi.util.write_bitmap(f"out/j{i}.png", img, write_async=False)

plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
