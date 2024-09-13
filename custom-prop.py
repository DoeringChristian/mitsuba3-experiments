import drjit as dr
import mitsuba as mi

if __name__ == "__main__":

    mi.set_variant("cuda_ad_rgb")

    scene = mi.cornell_box()
    scene = mi.load_dict(scene)

    dynamic_shape = scene.shapes()[0]
    dynamic_shape.set_id("dynamic_back")

    def is_dynamic(self, _) -> mi.Bool:
        if "dynamic" in self.id():
            print("return true")
            return mi.Bool(True)
        return mi.Bool(False)

    bsdf_ptrs = scene.shapes_dr()

    is_dynamic = dr.dispatch(bsdf_ptrs, is_dynamic, None)

    print(f"{is_dynamic=}")
