import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_log_level(dr.LogLevel.Trace)

sampler: mi.Sampler = mi.load_dict({"type": "independent"})
sampler.seed(0, 10)

depth = mi.UInt(0, 0, 0)

loop = mi.Loop("test", lambda: (depth))
while loop(depth < 10):
    depth += 1

print(f"{depth=}")
