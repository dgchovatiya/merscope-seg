import os
import re
import glob

import numpy as np
import tifffile

_FN = re.compile(r"mosaic_(?P<ch>[A-Za-z0-9]+)_z(?P<z>\d+)\.tif$")


class Region:
    """One imaged region: a directory of mosaic_<CH>_z<N>.tif files.

    Never loads pixels at construction; memmaps are opened lazily and cached.
    """

    def __init__(self, root: str, name: str | None = None):
        self.root = os.path.abspath(root)
        self.name = name or os.path.basename(self.root.rstrip("/"))
        self.files: dict[str, dict[int, str]] = {}
        for p in glob.glob(os.path.join(self.root, "mosaic_*_z*.tif")):
            m = _FN.search(os.path.basename(p))
            if m:
                self.files.setdefault(m["ch"], {})[int(m["z"])] = p
        if not self.files:
            raise FileNotFoundError(f"no mosaic_*_z*.tif under {self.root}")
        any_path = next(iter(next(iter(self.files.values())).values()))
        with tifffile.TiffFile(any_path) as t:
            page = t.pages.first
            self.shape = tuple(page.shape)          # (H, W)
            self.dtype = np.dtype(page.dtype)
        self._maps: dict[tuple, np.memmap] = {}
        self._thumbs: dict[tuple, np.ndarray] = {}

    # -- introspection -------------------------------------------------
    @property
    def channels(self) -> list[str]:
        return sorted(self.files)

    def z_planes(self, channel: str) -> list[int]:
        return sorted(self.files[channel])

    # -- pixel access --------------------------------------------------
    def memmap(self, channel: str, z: int) -> np.memmap:
        key = (channel, z)
        if key not in self._maps:
            self._maps[key] = tifffile.memmap(self.files[channel][z], mode="r")
        return self._maps[key]

    def read_window(self, channel: str, z: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        """Copy one window. Out-of-image parts (window may overhang the edge
        by design, e.g. margin reads) come back as zeros."""
        H, W = self.shape
        out = np.zeros((h, w), dtype=self.dtype)
        y0, x0 = max(y, 0), max(x, 0)
        y1, x1 = min(y + h, H), min(x + w, W)
        if y1 > y0 and x1 > x0:
            out[y0 - y : y1 - y, x0 - x : x1 - x] = self.memmap(channel, z)[y0:y1, x0:x1]
        return out

    def thumbnail(self, channel: str, z: int | None = None, stride: int = 32) -> np.ndarray:
        """Strided overview (float32). z=None -> middle plane. Cached."""
        zs = self.z_planes(channel)
        z = zs[len(zs) // 2] if z is None else z
        key = (channel, z, stride)
        if key not in self._thumbs:
            self._thumbs[key] = np.asarray(
                self.memmap(channel, z)[::stride, ::stride], dtype=np.float32
            ).copy()
        return self._thumbs[key]


class Catalog:
    """Finds all regions under the data root."""

    def __init__(self, data_root: str = "/Users/Projects/Data"):
        self.data_root = data_root
        self.regions: dict[str, Region] = {}
        for p in sorted(glob.glob(os.path.join(data_root, "retina", "Region *"))):
            r = Region(p, name="retina/" + os.path.basename(p))
            self.regions[r.name] = r
        brain = os.path.join(data_root, "Brain")
        if glob.glob(os.path.join(brain, "mosaic_*_z*.tif")):
            r = Region(brain, name="Brain")
            self.regions[r.name] = r

    def __getitem__(self, name: str) -> Region:
        return self.regions[name]

    def __iter__(self):
        return iter(self.regions.values())
