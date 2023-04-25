import gc
from typing import overload
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mi.set_variant("cuda_ad_rgb")
dr.set_log_level(dr.LogLevel.Debug)
mi.set_log_level(mi.LogLevel.Debug)
import pssmltpath
import pssmltsimple
import simple
import pssmlt


if __name__ == "__main__":
    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = 1024
    scene["sensor"]["film"]["height"] = 1024
    scene["sphere"] = {
        "type": "sphere",
        "to_world": mi.ScalarTransform4f.translate([0.335, -0.6, 0.38]).scale(0.3),
        "bsdf": {"type": "dielectric"},
    }
    # scene["blocking"] = {
    #     "type": "cube",
    #     "to_world": mi.ScalarTransform4f.translate([0.0, 0.4, 0.0]).scale(0.3),
    # }
    del scene["small-box"]
    print(f"{scene=}")
    scene = mi.load_dict(scene)
    scene = mi.load_file("data/veach-ajar/scene.xml")

    def render_pssmlt(integrator: pssmlt.Pssmlt, n=100, seed=0, mlt_burnin=50):
        # integrator: pssmltsimple.Pssmlt = mi.load_dict(
        #     {
        #         "type": "pssmlt",
        #         "max_depth": 8,
        #         "rr_depth": 2,
        #     }
        # )
        img = None
        nimg = None
        for i in range(n):
            print(f"{i=}")
            # nimg = mi.render(scene, integrator=integrator, seed=i + seed * n, spp=1)
            nimg = integrator.render(
                scene, scene.sensors()[0], seed=i + seed * n, spp=8
            )
            if img is None or i < mlt_burnin:
                img = nimg
            else:
                img = img * mi.Float(
                    (i - mlt_burnin) / (i - mlt_burnin + 1)
                ) + nimg / mi.Float(i - mlt_burnin + 1)
            # mi.util.write_bitmap(f"out/j{seed}i{i}.png", img, write_async=True)
        mi.util.write_bitmap(f"out/j{seed}-nacc.png", nimg)
        return img

    img = None
    with dr.suspend_grad():
        ref_integrator = mi.load_dict(
            {
                "type": "ptracer",
            }
        )
        ref = mi.render(scene, integrator=ref_integrator, spp=128)
        mi.util.write_bitmap("out/ref_ptrace.png", ref)
        del ref
        del ref_integrator
        gc.collect()
        dr.flush_malloc_cache()

        ref_integrator = mi.load_dict(
            {
                "type": "path",
            }
        )
        ref = mi.render(scene, integrator=ref_integrator, spp=128)
        mi.util.write_bitmap("out/ref.png", ref)
        del ref_integrator

        integrator: pssmltsimple.Pssmlt = mi.load_dict(
            {
                "type": "pssmlt",
                "max_depth": 8,
                "rr_depth": 2,
            }
        )

        for j in range(10):
            nimg = render_pssmlt(integrator, seed=j, n=100, mlt_burnin=50)

            if img is None:
                img = nimg
            else:
                img = img * mi.Float((j) / (j + 1)) + nimg / mi.Float(j + 1)
            mi.util.write_bitmap(f"out/j{j}.png", img, write_async=True)
            del nimg
            gc.collect()
            integrator.reset()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0][0].imshow(mi.util.convert_to_bitmap(img))
    axs[0][1].imshow(mi.util.convert_to_bitmap(ref))
    # plt.imshow(mi.util.convert_to_bitmap(img))
    plt.show()
