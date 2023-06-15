import mitsuba as mi
from tqdm import tqdm

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

import restirgi

if __name__ == "__main__":
    integrator: restirgi.RestirReservoir = mi.load_dict(
        {
            "type": "restirgi",
            "jacobian": True,
            "spatial_biased": True,
            "bsdf_sampling": True,
            "max_M_spatial": 500,
            "max_M_temporal": 30,
        }
    )

    scene: mi.Scene = mi.load_file("data/scenes/staircase/scene.xml")
    params = mi.traverse(scene)
    print(f"{params=}")

    for i in tqdm(range(200)):
        params["PerspectiveCamera.to_world"] @= mi.Transform4f.translate(
            [0.0, 0.0, 0.01]
        )
        params.update()

        img = mi.render(scene, params, seed=i, integrator=integrator, spp=1)
        mi.util.write_bitmap(f"out/{i}.jpg", img)
