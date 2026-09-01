"""mseg — MERSCOPE segmentation pipeline (Phase 1: reader + preprocessing).

Canonical plan: /Users/Projects/Docs/project-plan.md
Rules enforced here:
  * preprocessing changes brightness only, NEVER pixel positions (asserted)
  * background estimated on tile + margin, then cropped (no per-tile seams)
  * annotations anchor to raw coordinates + a recipe_id for provenance
"""
from .config import PreprocessConfig
from .regions import Region, Catalog
from .preprocess import Preprocessor
