"""Build the annotation package for Vincent.

Selects tiles (train: R0+R1, test: R2), preprocesses, runs Cellpose drafts,
converts masks -> VIA polygons, writes:
  out/annotation_package/
    via.html                     the browser annotator (offline, no install)
    tiles/*.png                  8-bit renderings for display
    drafts_annotations_via.json  draft polygons, preloaded-importable
    anchors.json                 raw-coordinate provenance per tile (OUR record)
    README.txt                   Vincent's instructions
"""
import os, sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image

from mseg import Catalog, Preprocessor, PreprocessConfig
from mseg.annot import mask_to_via_regions, write_via_json

N = 512                                   # dense retina -> small tiles
PLAN = {"retina/Region 0": ("train", 12),
        "retina/Region 1": ("train", 16),
        "retina/Region 2": ("test", 12)}  # Region 3 stays sealed
PILOT = 3                                 # first 3 train tiles = pilot set

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "annotation_package")
os.makedirs(os.path.join(OUT, "tiles"), exist_ok=True)

def pick(pp, region, n):
    """Stratified: sort tissue tiles by fraction, take a spread from dense to
    sparse, enforcing spatial separation."""
    H, W = region.shape
    cands = []
    for y in range(0, H - N, N):
        for x in range(0, W - N, N):
            f = pp.tissue_fraction(y, x, N, N)
            if f >= 0.10:
                cands.append((f, y, x))
    cands.sort(reverse=True)
    picked, want = [], np.linspace(0, min(len(cands), 400) - 1, n * 3).astype(int)
    for i in want:
        f, y, x = cands[i]
        if all(abs(y - py) + abs(x - px) > 2 * N for _, py, px in picked):
            picked.append((f, y, x))
        if len(picked) == n:
            break
    return picked

def main():
    import torch
    from cellpose import models
    model = models.CellposeModel(gpu=True, device=torch.device("mps"))
    cfg = PreprocessConfig()
    cat = Catalog()
    entries, anchors = [], []
    idx = 0
    for rname, (role, n) in PLAN.items():
        region = cat[rname]
        pp = Preprocessor(region, "DAPI", cfg)
        tiles = pick(pp, region, n)
        print(f"{rname}: {len(tiles)} tiles ({role})", flush=True)
        for f, y, x in tiles:
            idx += 1
            t0 = time.time()
            img = pp.process(y, x, N, N)
            masks, _, _ = model.eval(img, batch_size=8, normalize=True)
            n_obj = int(masks.max())
            fn = f"{idx:03d}_{rname.split('/')[-1].replace(' ', '')}_y{y}_x{x}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(
                os.path.join(OUT, "tiles", fn))
            size = os.path.getsize(os.path.join(OUT, "tiles", fn))
            entries.append(dict(
                filename=fn, size=size,
                file_attributes=dict(height=N, width=N, role=role,
                                     pilot=bool(idx <= PILOT)),
                regions=mask_to_via_regions(masks, min_area=60),
            ))
            anchors.append(dict(filename=fn, role=role, pilot=bool(idx <= PILOT),
                                draft_objects=n_obj, tissue_fraction=round(f, 2),
                                **pp.anchor(y, x, N, N)))
            print(f"  {fn}  tissue={f:.2f}  drafts={n_obj}  {time.time()-t0:.1f}s",
                  flush=True)
    write_via_json(os.path.join(OUT, "drafts_annotations_via.json"), entries)
    json.dump(anchors, open(os.path.join(OUT, "anchors.json"), "w"), indent=1)
    total = sum(a["draft_objects"] for a in anchors)
    print(f"\nPACKAGE: {len(anchors)} tiles, {total} draft objects", flush=True)

if __name__ == "__main__":
    main()
