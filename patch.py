import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

"""
Proof of concept implementation of automated monkeypatching of mitsuba vcalls.
"""

if __name__ == "__main__":
    mi.set_variant("cuda_ad_spectral")
    dr.set_flag(dr.JitFlag.KernelHistory, True)


def patch(*functions: list[str]):
    def patch_decorator(cls):
        import mitsuba as mi

        assert len(cls.__bases__) == 1

        base = cls.__bases__[0]

        ptr_class = getattr(mi, f"{base.__name__}Ptr")

        for function in functions:

            if not hasattr(base, function):

                def default_impl(self, *args, **kwargs):
                    raise RuntimeError("Method not implemented!")

                setattr(base, function, default_impl)

            def _impl(self, *args, **kwargs):
                if hasattr(self, "sample_wavelength"):
                    return self.sample_wavelength(*args, **kwargs)

            def dispatch_impl(self, *args, **kwargs):
                return dr.dispatch(self, _impl, *args, **kwargs)

            setattr(ptr_class, function, dispatch_impl)

        return cls

    return patch_decorator


def sample_wavelength(self, wavelengths: mi.Spectrum) -> mi.Spectrum:
    print(f"default {self.id()}")
    return wavelengths


# Add default implementation
mi.BSDF.sample_wavelength = sample_wavelength


@patch("sample_wavelength")
class Flouresent(mi.BSDF):
    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)

    def sample_wavelength(self, wavelengths: mi.Spectrum) -> mi.Spectrum:
        print(f"flouresent {self.id()}")
        return wavelengths + 1

    def to_string(self):
        return "Flouresent[]"


mi.register_bsdf("flouresent", lambda props: Flouresent(props))

if __name__ == "__main__":
    scene = mi.cornell_box()
    scene["white"] = {"type": "flouresent"}
    scene: mi.Scene = mi.load_dict(scene)

    shape = scene.shapes_dr()
    bsdf = shape.bsdf()

    wavelengths = mi.Spectrum(0)

    # dr.dispatch(bsdf, sample_wavelength, wavelengths)

    result = bsdf.sample_wavelength(wavelengths)
    print(f"{result=}")
    for shape in scene.shapes():
        print(f"{shape.bsdf().id()=}")
