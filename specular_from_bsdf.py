import drjit as dr
import mitsuba as mi
from typing import Tuple

mi.set_variant("cuda_ad_rgb")


class DiffuseBSDF(mi.BSDF):
    def __init__(self: mi.BSDF, props: mi.Properties) -> None:
        super().__init__(props)

        self.reflectance = props.get("reflectance")

        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def sample(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ) -> Tuple[mi.BSDFSample3f, mi.Color3f]:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.DiffuseReflection
        bs.sampled_component = 0

        value = self.reflectance.eval(si, active)

        return (bs, value)

    def eval(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> mi.Color3f:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        value = self.reflectance.eval(si, active) * dr.inv_pi * cos_theta_o

        return value * dr.select(active, 1.0, 0.0)

    def pdf(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> float:

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def specular(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        return mi.Color3f(0.0)


class SpecularBSDF(mi.BSDF):
    def __init__(self: mi.BSDF, props: mi.Properties) -> None:
        super().__init__(props)

        self.reflectance = props.get("reflectance")

        self.m_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def sample(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ) -> Tuple[mi.BSDFSample3f, mi.Color3f]:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        bs = mi.BSDFSample3f()
        bs.wo = mi.reflect(si.wi)
        bs.pdf = 1.0
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.DeltaReflection
        bs.sampled_component = 0

        value = self.reflectance.eval(si, active)

        return (bs, value)

    def eval(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> mi.Color3f:
        return 0.0

    def pdf(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> float:

        return 0.0

    def specular(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        return self.reflectance.eval(si)


mi.register_bsdf("diffuse", lambda props: DiffuseBSDF(props))
mi.register_bsdf("conductor", lambda props: SpecularBSDF(props))

if __name__ == "__main__":

    scene = mi.cornell_box()
    scene["red"]["type"] = "conductor"
    scene = mi.load_dict(scene)

    img = mi.render(scene)

    mi.util.write_bitmap("out/test.jpg", img)

    def specular(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        return self.specular(si)

    bsdf_ptrs = scene.shapes_dr().bsdf()
    print(f"{dr.width(bsdf_ptrs)=}")

    si = dr.zeros(mi.SurfaceInteraction3f, 8)
    specular_color = dr.dispatch(bsdf_ptrs, specular, si)
    print(f"{specular_color=}")
