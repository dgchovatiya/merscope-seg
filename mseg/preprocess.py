import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

from .config import PreprocessConfig
from .regions import Region


class Preprocessor:
    """Turns raw z-stack windows into model-ready float32 images in [0, 1].

    Contract: output shape == requested shape, pixel positions untouched
    (brightness-only). Enforced by assertion on every call.
    """

    def __init__(self, region: Region, channel: str, cfg: PreprocessConfig | None = None):
        self.region = region
        self.channel = channel
        self.cfg = cfg or PreprocessConfig()
        self._tissue_thumb = None
        self._tissue_thr = None

    # -- plane selection ----------------------------------------------
    def planes(self) -> list[int]:
        avail = self.region.z_planes(self.channel)
        zp = self.cfg.z_planes
        if zp == "all":
            return avail
        want = [z for z in zp if z in avail]
        if not want:
            raise ValueError(f"none of z={zp} available (have {avail})")
        return want

    # -- stages (each is brightness-only) ------------------------------
    def _project(self, y, x, h, w) -> np.ndarray:
        zs = self.planes()
        if self.cfg.projection == "single":
            zs = zs[:1]
        acc = None
        for z in zs:
            plane = self.region.read_window(self.channel, z, y, x, h, w).astype(np.float32)
            if acc is None:
                acc = plane
            elif self.cfg.projection == "max":
                np.maximum(acc, plane, out=acc)
            elif self.cfg.projection == "mean":
                acc += plane
            else:
                raise ValueError(self.cfg.projection)
        if self.cfg.projection == "mean":
            acc /= len(zs)
        return acc

    @staticmethod
    def _destripe(img: np.ndarray) -> np.ndarray:
        """Experimental: flatten smooth row/column offset (FOV stitching seams).
        Median profile minus its own smooth trend = the stripe component."""
        out = img
        for axis in (0, 1):
            prof = np.median(out, axis=axis)
            trend = gaussian_filter(prof, 25.0)
            stripe = prof - trend
            out = out - (stripe[None, :] if axis == 0 else stripe[:, None])
        return out

    # -- main entry ----------------------------------------------------
    def process(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        """Model-ready window. Background is estimated on tile+margin then
        cropped back, so neighbouring tiles get consistent estimates."""
        m = self.cfg.margin_px()
        img = self._project(y - m, x - m, h + 2 * m, w + 2 * m)
        if self.cfg.destripe:
            img = self._destripe(img)
        if self.cfg.bg_sigma:
            img = img - gaussian_filter(img, self.cfg.bg_sigma)
        if m:
            img = img[m:-m, m:-m]
        lo, hi = np.percentile(img, self.cfg.clip_pct)
        img = np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        assert img.shape == (h, w), f"geometry changed: {img.shape} != {(h, w)}"
        return img.astype(np.float32, copy=False)

    def raw(self, y: int, x: int, h: int, w: int, z: int | None = None) -> np.ndarray:
        """Un-preprocessed single plane (display/comparison)."""
        zs = self.region.z_planes(self.channel)
        z = zs[len(zs) // 2] if z is None else z
        return self.region.read_window(self.channel, z, y, x, h, w)

    # -- tissue gating (thumbnail-based, cheap) ------------------------
    def _ensure_tissue_thumb(self, stride=32):
        if self._tissue_thumb is None:
            t = self.region.thumbnail(self.channel, stride=stride)
            try:
                thr = float(threshold_otsu(t))
            except ValueError:            # constant image
                thr = float(t.max()) + 1.0
            self._tissue_thr = thr
            self._tissue_thumb = (t > self._tissue_thr)
            self._tissue_stride = stride

    def tissue_fraction(self, y: int, x: int, h: int, w: int) -> float:
        self._ensure_tissue_thumb()
        s = self._tissue_stride
        block = self._tissue_thumb[y // s : (y + h) // s + 1, x // s : (x + w) // s + 1]
        return float(block.mean()) if block.size else 0.0

    def is_tissue(self, y, x, h, w, min_fraction: float = 0.05) -> bool:
        return self.tissue_fraction(y, x, h, w) >= min_fraction

    # -- provenance ----------------------------------------------------
    def anchor(self, y: int, x: int, h: int, w: int) -> dict:
        """The record stored WITH annotations: everything needed to re-render
        this exact tile from raw under any future recipe."""
        return dict(
            region=self.region.name, channel=self.channel,
            y=int(y), x=int(x), h=int(h), w=int(w),
            z_planes=list(self.planes()), projection=self.cfg.projection,
            recipe_id=self.cfg.recipe_id(),
        )
