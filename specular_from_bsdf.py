import drjit as dr
import mitsuba as mi
from typing import Tuple

mi.set_variant("cuda_ad_rgb")


if __name__ == "__main__":

    scene = mi.cornell_box()
    scene["red"]["type"] = "conductor"
    scene["red"] = {
        "type": "dielectric",
        "specular_reflectance": {
            "type": "bitmap",
            "filename": "WoodFloor.jpg",
        },
    }
    scene = mi.load_dict(scene)

    img = mi.render(scene)

    mi.util.write_bitmap("out/test.jpg", img)

    def specular(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        params = mi.traverse(self)

        sr = "specular_reflectance"
        if f"{sr}.data" in params and f"{sr}.to_uv":
            data = params[f"{sr}.data"]
            to_uv = params[f"{sr}.to_uv"]

            texture = mi.Texture2f(data)

            to_uv = mi.Transform3f(to_uv)

            uv = to_uv @ si.uv

            return mi.Color3f(texture.eval(uv))
        sr = "specular_reflectance"
        if f"{sr}.value" in params:
            return mi.Color3f(params[f"{sr}.value"])
        else:
            return mi.Color3f(0)

    bsdf_ptrs = scene.shapes_dr().bsdf()
    print(f"{dr.width(bsdf_ptrs)=}")

    si = dr.zeros(mi.SurfaceInteraction3f, 8)
    specular_color = dr.dispatch(bsdf_ptrs, specular, si)
    print(f"{specular_color=}")
