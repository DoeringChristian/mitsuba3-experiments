import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def to_dict(scene: mi.Scene):
    assert isinstance(scene, mi.Scene)

    unknown_counter = 0

    def get_id(child: mi.Object):
        nonlocal unknown_counter
        id = child.id()
        if id == "":
            id = f"unknown{unknown_counter}"
            unknown_counter += 1
        return id

    children = [
        *scene.shapes(),
        *scene.emitters(),
        *scene.sensors(),
        scene.integrator(),
    ]
    return {
        "type": "scene",
        **{get_id(child): child for child in children},
    }


if __name__ == "__main__":
    scene = mi.cornell_box()
    scene["sg"] = {
        "type": "shapegroup",
        "second_object": {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f()
            .translate([-0.5, 0, 0])
            .scale([0.2, 0.2, 0.2]),
            "bsdf": {
                "type": "diffuse",
            },
        },
    }
    scene["first_instance"] = {
        "type": "instance",
        "shapegroup": {"type": "ref", "id": "sg"},
    }

    scene = mi.load_dict(scene)

    ref = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/ref.exr", ref)

    scene = to_dict(scene)
    print(f"{scene=}")

    scene = mi.load_dict(scene)

    res = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/res.exr", ref)

    assert dr.allclose(ref, res)
