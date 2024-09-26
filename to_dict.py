import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def to_dict(obj: mi.Object):
    id = obj.id()
    tp = obj.class_().name()

    if tp == "Scene":
        children = [
            *scene.shapes(),
            *scene.emitters(),
            *scene.sensors(),
            scene.integrator(),
        ]
        return {
            "type": "scene",
            **{child.id(): to_dict(child) for child in children},
        }
    else:
        return obj


if __name__ == "__main__":

    scene = mi.load_dict(mi.cornell_box())

    ref = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/ref.exr", ref)

    scene = to_dict(scene)

    scene = mi.load_dict(scene)

    res = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/res.exr", ref)

    assert dr.allclose(ref, res)
