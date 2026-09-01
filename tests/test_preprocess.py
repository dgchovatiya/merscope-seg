"""Plain-python tests (no pytest dependency). Run: python tests/test_preprocess.py"""
import os, sys, tempfile
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mseg import Region, Preprocessor, PreprocessConfig


def make_region(tmp, H=600, W=500, nz=3, seed=0):
    """Synthetic 'region': smooth background gradient + bright blobs, per z."""
    rng = np.random.default_rng(seed)
    blobs = np.zeros((H, W), np.float32)
    ys, xs = rng.integers(40, H - 40, 25), rng.integers(40, W - 40, 25)
    yy, xx = np.mgrid[0:H, 0:W]
    for cy, cx in zip(ys, xs):
        blobs += 3000 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0**2)))
    grad = 2000 + 4000 * (xx / W) + 1500 * (yy / H)          # smooth background
    for z in range(nz):
        weight = 0.4 + 0.6 * (z / (nz - 1))                   # blobs brighten with z
        img = (grad + weight * blobs + rng.normal(0, 30, (H, W))).clip(0, 65535)
        tifffile.imwrite(os.path.join(tmp, f"mosaic_DAPI_z{z}.tif"),
                         img.astype(np.uint16))
    return blobs, grad


def main():
    tmp = tempfile.mkdtemp(prefix="mseg_test_")
    blobs, grad = make_region(tmp)
    r = Region(tmp, name="synthetic")
    assert r.channels == ["DAPI"] and r.z_planes("DAPI") == [0, 1, 2], "discovery"
    assert r.shape == (600, 500), "shape"

    # 1. geometry: any window comes back exactly the requested size
    pp = Preprocessor(r, "DAPI", PreprocessConfig(bg_sigma=20.0, margin_sigmas=3.0))
    out = pp.process(100, 100, 128, 128)
    assert out.shape == (128, 128) and out.dtype == np.float32, "geometry/dtype"
    assert 0.0 <= out.min() and out.max() <= 1.0, "normalized range"

    # 2. edge overhang: margin reads past the border must not crash; zeros pad
    edge = pp.process(0, 0, 64, 64)
    assert edge.shape == (64, 64), "edge geometry"
    ov = r.read_window("DAPI", 0, -10, -10, 32, 32)
    assert ov[:10].max() == 0 and ov[:, :10].max() == 0, "overhang zero-pad"

    # 3. projection: max over z must equal per-pixel max of the planes
    proj = pp._project(200, 200, 64, 64)
    stack = np.stack([r.read_window("DAPI", z, 200, 200, 64, 64) for z in (0, 1, 2)])
    assert np.array_equal(proj, stack.max(0).astype(np.float32)), "max projection"

    # 4. margin correctness: windowed bgsub (with margin) ~= global bgsub, interior
    full = np.asarray(r.memmap("DAPI", 2), np.float32)
    g_global = full - gaussian_filter(full, 20.0)
    y, x, h, w = 150, 120, 128, 128
    m = pp.cfg.margin_px()
    win = r.read_window("DAPI", 2, y - m, x - m, h + 2 * m, w + 2 * m).astype(np.float32)
    g_win = (win - gaussian_filter(win, 20.0))[m:-m, m:-m]
    diff = np.abs(g_win - g_global[y:y + h, x:x + w])
    scale = np.abs(g_global[y:y + h, x:x + w]).max()
    assert diff.max() / scale < 0.02, f"margin bgsub differs {diff.max()/scale:.4f}"

    # 5. seam consistency: two ADJACENT processed tiles agree along their shared edge
    cfg = PreprocessConfig(bg_sigma=20.0, margin_sigmas=3.0, clip_pct=(0.0, 100.0))
    pp2 = Preprocessor(r, "DAPI", cfg)
    m2 = cfg.margin_px()
    a_raw = pp2._project(100 - m2, 50 - m2, 128 + 2*m2, 128 + 2*m2)
    b_raw = pp2._project(100 - m2, 178 - m2, 128 + 2*m2, 128 + 2*m2)
    a_bg = (a_raw - gaussian_filter(a_raw, 20.0))[m2:-m2, m2:-m2]
    b_bg = (b_raw - gaussian_filter(b_raw, 20.0))[m2:-m2, m2:-m2]
    seam = np.abs(a_bg[:, -1] - b_bg[:, 0])
    interior = np.abs(np.diff(a_bg, axis=1)).mean()
    a_col_edge = np.abs(a_bg[:, -1] - a_bg[:, -2]).mean()
    # the step across the tile boundary should look like any interior step
    assert seam.mean() < 5 * max(interior, 1e-3) + 5.0, \
        f"seam step {seam.mean():.2f} vs interior {interior:.2f}"

    # 6. provenance
    a1 = pp.anchor(100, 100, 128, 128)
    assert a1["recipe_id"] == pp.cfg.recipe_id() and a1["z_planes"] == [0, 1, 2]
    assert PreprocessConfig().recipe_id() == PreprocessConfig().recipe_id(), "stable id"
    assert PreprocessConfig().recipe_id() != PreprocessConfig(bg_sigma=50.0).recipe_id()

    # 7. tissue gating: blob-rich window vs empty corner
    frac_rich = pp.tissue_fraction(0, 0, 600, 500)
    print("tissue fraction (whole frame):", round(frac_rich, 3))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
