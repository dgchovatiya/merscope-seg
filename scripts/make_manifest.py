"""Inventory every region; write out/manifest.json. Header-only + one strided
thumbnail per region — no full-image reads."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mseg import Catalog, Preprocessor, PreprocessConfig

cat = Catalog()
cfg = PreprocessConfig()
rows = []
for r in cat:
    t0 = time.time()
    pp = Preprocessor(r, "DAPI", cfg)
    frac = pp.tissue_fraction(0, 0, *r.shape)
    rows.append(dict(
        name=r.name, shape=list(r.shape), gigapixels=round(r.shape[0]*r.shape[1]/1e9, 2),
        channels={c: r.z_planes(c) for c in r.channels},
        dtype=str(r.dtype), tissue_fraction=round(frac, 3),
        scan_s=round(time.time() - t0, 1),
    ))
    print(f"{r.name:20s} {r.shape[1]:>6}x{r.shape[0]:<6} tissue={frac:.1%} ({rows[-1]['scan_s']}s)")
json.dump(dict(recipe_id=cfg.recipe_id(), regions=rows),
          open(os.path.join(os.path.dirname(__file__), "..", "out", "manifest.json"), "w"), indent=1)
print("wrote out/manifest.json  recipe:", cfg.recipe_id())
