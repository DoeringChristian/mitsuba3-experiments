from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")
import pssmltpath
import pssmltsimple


scene = mi.cornell_box()
integrator = mi.load_dict(
    {
        "type": "pssmlt",
        "max_depth": 8,
        "rr_depth": 2,
    }
)
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

img = None
mlt_depth = 50
j = 1
with dr.suspend_grad():
    for i in range(100):
        print(f"{i=}")
        nimg = mi.render(scene, integrator=integrator, seed=i, spp=1)
        if i < mlt_depth:
            img = nimg
        else:
            img = img * mi.Float((j - 1) / j) + nimg / mi.Float(j)
            j += 1
        mi.util.write_bitmap(f"out/{i}.png", img, write_async=False)

plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()
