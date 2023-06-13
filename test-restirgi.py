import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

import restirgi

if __name__ == "__main__":
    n_iterations = 100
    spp = 4

    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = 1024
    scene["sensor"]["film"]["height"] = 1024
    scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
    scene: mi.Scene = mi.load_dict(scene)

    ref = mi.render(scene, spp=50 * 4)

    jacobian: restirgi.RestirIntegrator = mi.load_dict(
        {
            "type": "restirgi",
            "jacobian": True,
            "spatial_biased": True,
            "bsdf_sampling": True,
            "max_M_spatial": 500,
            "max_M_temporal": 30,
        }
    )

    nonjacobian: restirgi.RestirIntegrator = mi.load_dict(
        {
            "type": "restirgi",
            "jacobian": False,
            "spatial_biased": True,
            "bsdf_sampling": True,
            "max_M_spatial": 500,
            "max_M_temporal": 30,
        }
    )

    var_jacobian = []
    bias_jacobian = []

    print("Jacobian Bias Corrected:")
    for i in tqdm(range(n_iterations)):
        img = mi.render(scene, integrator=jacobian, seed=i, spp=spp)
        var_jacobian.append(dr.mean_nested(dr.sqr(img - dr.mean_nested(img)))[0])
        bias_jacobian.append(dr.mean_nested(ref - img)[0])

    img_jacobian = img

    var_nonjacobian = []
    bias_nonjacobian = []

    print("Non Jacobian Bias Corrected:")
    for i in tqdm(range(n_iterations)):
        img = mi.render(scene, integrator=nonjacobian, seed=i, spp=spp)
        var_nonjacobian.append(dr.mean_nested(dr.sqr(img - dr.mean_nested(img)))[0])
        bias_nonjacobian.append(dr.mean_nested(ref - img)[0])

    img_nonjacobian = img

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_visible(False)

    ax[0][0].axis("off")
    ax[0][0].imshow(mi.util.convert_to_bitmap(ref))
    ax[0][0].set_title("ref")

    ax[0][1].plot(bias_jacobian, label="Jacobian Bias Correction")
    ax[0][1].plot(bias_nonjacobian, label="No Jacobian Bias Correction")
    ax[0][1].legend(loc="best")

    ax[1][0].axis("off")
    ax[1][0].set_title("No Jacobian Bias Correction")
    ax[1][0].imshow(mi.util.convert_to_bitmap(img_nonjacobian))

    ax[1][1].axis("off")
    ax[1][1].set_title("Jacobian Bias Correction")
    ax[1][1].imshow(mi.util.convert_to_bitmap(img_jacobian))

    fig.tight_layout()
    plt.show()
