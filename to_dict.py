import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def to_dict(scene: mi.Scene) -> dict:
    result = {
        "type": "scene",
    }

    for shape in scene.shapes():
        id = shape.id()
        result[id] = shape

    for emitter in scene.emitters():
        id = emitter.id()
        result[id] = emitter

    for sensor in scene.sensors():
        id = sensor.id()
        result[id] = sensor

    id = scene.integrator().id()
    result[id] = scene.integrator()

    return result


if __name__ == "__main__":

    scene = mi.load_dict(mi.cornell_box())

    ref = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/ref.exr", ref)

    scene = to_dict(scene)

    scene = mi.load_dict(scene)

    res = mi.render(scene, spp=128)
    mi.util.write_bitmap("out/res.exr", ref)

    assert dr.allclose(ref, res)
