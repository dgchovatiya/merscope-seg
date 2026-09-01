"""Annotation format converters.

Formats:
  * label mask   — uint16 array, 0 = background, N = object id (model native)
  * VIA JSON     — polygon annotations, the format Vincent edits in the browser
                   and EXACTLY what Andrei's rpe_dataset.py ingests
  * RLE CSV      — Andrei's interchange: rows of (ID, Frame, y, xL, xR)

Annotations are coordinates; images are just renderings (see project plan).
"""
import json
import numpy as np
from skimage import measure, draw


# ---------- mask -> VIA ------------------------------------------------
def mask_to_via_regions(mask: np.ndarray, tolerance: float = 1.5,
                        min_area: int = 30) -> list[dict]:
    """One VIA polygon region per labelled object."""
    regions = []
    for lab in np.unique(mask):
        if lab == 0:
            continue
        binary = mask == lab
        if binary.sum() < min_area:
            continue
        padded = np.pad(binary, 1)                       # close contours at edges
        for cont in measure.find_contours(padded.astype(float), 0.5):
            cont = measure.approximate_polygon(cont, tolerance) - 1.0
            if len(cont) < 3:
                continue
            ys = np.clip(cont[:, 0], 0, mask.shape[0] - 1)
            xs = np.clip(cont[:, 1], 0, mask.shape[1] - 1)
            regions.append({
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [int(round(v)) for v in xs],
                    "all_points_y": [int(round(v)) for v in ys],
                },
                "region_attributes": {"draft_id": int(lab)},
            })
            break                                        # outer contour only
    return regions


def write_via_json(path: str, entries: list[dict]) -> None:
    """entries: [{filename, size, file_attributes, regions}] -> VIA dict file
    keyed filename+size, the shape Andrei's loader reads."""
    doc = {}
    for e in entries:
        doc[e["filename"] + str(e["size"])] = {
            "filename": e["filename"], "size": e["size"],
            "file_attributes": e.get("file_attributes", {}),
            "regions": e["regions"],
        }
    with open(path, "w") as fo:
        json.dump(doc, fo)


# ---------- VIA -> masks ----------------------------------------------
def via_to_masks(via_doc: dict, height: int, width: int,
                 filename: str | None = None) -> np.ndarray:
    """Rasterize one file's polygons back to a label mask."""
    items = [v for v in via_doc.values()
             if filename is None or v["filename"] == filename]
    if not items:
        raise KeyError(f"{filename} not in VIA doc")
    regions = items[0]["regions"]
    if isinstance(regions, dict):                        # VIA sometimes dict-keyed
        regions = list(regions.values())
    mask = np.zeros((height, width), np.uint16)
    for i, reg in enumerate(regions, start=1):
        sa = reg["shape_attributes"]
        rr, cc = draw.polygon(sa["all_points_y"], sa["all_points_x"],
                              shape=(height, width))
        mask[rr, cc] = i
    return mask


# ---------- mask <-> Andrei's RLE CSV ---------------------------------
def mask_to_rle_rows(mask: np.ndarray, frame: int = 0) -> list[tuple]:
    """(ID, Frame, y, xL, xR) inclusive runs, matching Andrei's format."""
    rows = []
    for y in range(mask.shape[0]):
        line = mask[y]
        edges = np.flatnonzero(np.diff(line) != 0) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(line)]))
        for s, e in zip(starts, ends):
            lab = int(line[s])
            if lab:
                rows.append((lab, frame, y, int(s), int(e - 1)))
    return rows


def rle_rows_to_mask(rows, height: int, width: int, frame: int = 0) -> np.ndarray:
    mask = np.zeros((height, width), np.uint16)
    for lab, fr, y, xl, xr in rows:
        if fr == frame:
            mask[y, xl:xr + 1] = lab
    return mask
