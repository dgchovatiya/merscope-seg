import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skimage import draw as skdraw
from mseg.annot import (mask_to_via_regions, write_via_json, via_to_masks,
                        mask_to_rle_rows, rle_rows_to_mask)

# synthetic mask: 3 blobs, one touching the edge
mask = np.zeros((256, 256), np.uint16)
for i, (cy, cx, r) in enumerate([(60, 60, 25), (150, 180, 30), (250, 20, 20)], 1):
    rr, cc = skdraw.disk((cy, cx), r, shape=mask.shape)
    mask[rr, cc] = i

# mask -> VIA -> mask round trip: per-object IoU must stay high
regions = mask_to_via_regions(mask)
assert len(regions) == 3, f"expected 3 regions, got {len(regions)}"
entries = [dict(filename="t.png", size=1234,
                file_attributes=dict(height=256, width=256), regions=regions)]
write_via_json("/tmp/_via_test.json", entries)
import json
back = via_to_masks(json.load(open("/tmp/_via_test.json")), 256, 256)
for lab in (1, 2, 3):
    a = mask == lab
    best = max(((back == l) for l in np.unique(back) if l),
               key=lambda b: (a & b).sum())
    iou = (a & best).sum() / (a | best).sum()
    assert iou > 0.93, f"object {lab} round-trip IoU {iou:.3f}"

# mask -> RLE -> mask must be EXACT
rows = mask_to_rle_rows(mask, frame=7)
assert all(r[1] == 7 for r in rows)
back2 = rle_rows_to_mask(rows, 256, 256, frame=7)
assert np.array_equal(mask, back2), "RLE round trip not exact"
print(f"ALL ANNOT TESTS PASSED  ({len(rows)} RLE runs, VIA IoU>0.93 x3)")
