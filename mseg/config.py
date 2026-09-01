import hashlib
import json
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class PreprocessConfig:
    """Recipe for turning raw z-stack windows into model-ready images.

    Brightness-only by contract: nothing here may move, resize or warp pixels.
    """
    z_planes: tuple | str = "all"      # "all", or explicit tuple like (2,3,4,5)
    projection: str = "max"            # max | mean | single (single uses z_planes[0])
    destripe: bool = False             # experimental FOV-seam correction
    bg_sigma: float | None = 60.0      # gaussian sigma (px) for background estimate; None = off
    margin_sigmas: float = 3.0         # margin read around tile = margin_sigmas * bg_sigma
    clip_pct: tuple = (1.0, 99.5)      # percentile clip for normalization

    def margin_px(self) -> int:
        if not self.bg_sigma:
            return 0
        return int(round(self.margin_sigmas * self.bg_sigma))

    def recipe_id(self) -> str:
        """Stable short id stored alongside annotations for provenance."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return "pp-" + hashlib.sha1(blob.encode()).hexdigest()[:10]
