import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

import restirgi

if __name__ == "__main__":
    n_iterations = 100
    spp = 1

    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = 1024
    scene["sensor"]["film"]["height"] = 1024
    scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
    scene: mi.Scene = mi.load_dict(scene)
    scene = mi.load_file("./data/scenes/staircase/scene.xml")

    ref = mi.render(scene, spp=50 * 4)

    biased: restirgi.RestirIntegrator = mi.load_dict(
        {
            "type": "restirgi",
            "jacobian": False,
            "spatial_biased": True,
            "bsdf_sampling": True,
            "max_M_spatial": 500,
            "max_M_temporal": 30,
        }
    )

    unbiased: restirgi.RestirIntegrator = mi.load_dict(
        {
            "type": "restirgi",
            "jacobian": False,
            "spatial_biased": False,
            "bsdf_sampling": True,
            "max_M_spatial": 500,
            "max_M_temporal": 30,
        }
    )

    var_biased = []
    bias_biased = []
    mse_biased = []

    print("Jacobian Bias Corrected:")
    for i in tqdm(range(n_iterations)):
        img = mi.render(scene, integrator=biased, seed=i, spp=spp)
        var_biased.append(dr.mean_nested(dr.sqr(img - dr.mean_nested(img)))[0])
        bias_biased.append(dr.abs(dr.mean_nested(ref - img))[0])
        mse_biased.append(dr.mean_nested(dr.sqr(img - ref)))

    img_biased = img

    var_unbiased = []
    bias_unbiased = []
    mse_unbiased = []

    print("Non Jacobian Bias Corrected:")
    for i in tqdm(range(n_iterations)):
        img = mi.render(scene, integrator=unbiased, seed=i, spp=spp)
        var_unbiased.append(dr.mean_nested(dr.sqr(img - dr.mean_nested(img)))[0])
        bias_unbiased.append(dr.abs(dr.mean_nested(ref - img))[0])
        mse_unbiased.append(dr.mean_nested(dr.sqr(img - ref)))

    img_unbiased = img

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_visible(False)

    ax[0][0].axis("off")
    ax[0][0].imshow(mi.util.convert_to_bitmap(ref))
    ax[0][0].set_title("ref")

    ax[0][1].plot(bias_biased, label="Biased")
    ax[0][1].plot(bias_unbiased, label="Bias Corrected")
    ax[0][1].legend(loc="best")

    ax[1][0].axis("off")
    ax[1][0].set_title("Bias Corrected")
    ax[1][0].imshow(mi.util.convert_to_bitmap(img_unbiased))

    ax[1][1].axis("off")
    ax[1][1].set_title("Biased")
    ax[1][1].imshow(mi.util.convert_to_bitmap(img_biased))

    fig.tight_layout()
    plt.show()
