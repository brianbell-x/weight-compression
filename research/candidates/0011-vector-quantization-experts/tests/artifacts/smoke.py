import time, torch
import vq_probe as v

t0 = time.time()
names = ["backbone.layers.1.mixer.experts.0.up_proj.weight"]
configs = [
    {"d": 8, "nbits": 8, "R": 2, "iter": 8},   # ~2.0 b/w
    {"d": 8, "nbits": 8, "R": 3, "iter": 8},   # ~3.0 b/w
]
rows = v.run(names, configs, use_incoherence_variants=True)
print(f"\nelapsed {time.time()-t0:.1f}s")
for r in rows:
    print(r)
