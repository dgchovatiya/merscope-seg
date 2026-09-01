"""Raw-vs-preprocessed comparison sheets for Vincent's eyeball test.
One PNG per retina region: 3 tissue tiles, [raw mid-z | preprocessed] pairs."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mseg import Catalog, Preprocessor, PreprocessConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
N = 1024

def pick_tiles(pp, region, n=3):
    """Grid-scan the thumbnail; pick tissue tiles spread across the ribbon."""
    H, W = region.shape
    cands = []
    for y in range(0, H - N, N):
        for x in range(0, W - N, N):
            f = pp.tissue_fraction(y, x, N, N)
            if 0.15 <= f <= 0.95:
                cands.append((f, y, x))
    cands.sort(reverse=True)
    picked = []
    for f, y, x in cands:
        if all(abs(y - py) + abs(x - px) > 4 * N for _, py, px in picked):
            picked.append((f, y, x))
        if len(picked) == n:
            break
    return picked

def stretch(a, lo=1, hi=99.5):
    a = a.astype(np.float32)
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-6), 0, 1)

cat = Catalog()
cfg = PreprocessConfig()
for r in cat:
    if not r.name.startswith("retina"):
        continue
    pp = Preprocessor(r, "DAPI", cfg)
    tiles = pick_tiles(pp, r)
    if not tiles:
        print(f"{r.name}: no tiles in range"); continue
    fig, axes = plt.subplots(len(tiles), 2, figsize=(11, 5.4 * len(tiles)),
                             facecolor="#0d0d10", squeeze=False)
    for i, (f, y, x) in enumerate(tiles):
        t0 = time.time()
        raw = pp.raw(y, x, N, N)
        proc = pp.process(y, x, N, N)
        dt = time.time() - t0
        axes[i][0].imshow(stretch(raw), cmap="gray", interpolation="nearest")
        axes[i][0].set_title(f"RAW  z=mid   y={y} x={x}", color="#9fb3c8", fontsize=10)
        axes[i][1].imshow(proc, cmap="gray", interpolation="nearest")
        axes[i][1].set_title(f"PREPROCESSED  (8z max-proj + bgsub)   {dt:.1f}s",
                             color="#3ef08c", fontsize=10)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        print(f"{r.name} tile y={y} x={x} tissue={f:.2f} {dt:.1f}s", flush=True)
    fig.suptitle(f"{r.name} — is anything visible in RAW that is missing in PREPROCESSED?",
                 color="white", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT, f"sheet_{r.name.replace('/', '_').replace(' ', '')}.png")
    fig.savefig(out, dpi=85, facecolor="#0d0d10")
    plt.close(fig)
    print("wrote", out, flush=True)
