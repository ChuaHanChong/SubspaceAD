"""Infiray IR ship dataset adapter.

Layout (under a dataset root):
    <split>/{0..6}/*.jpg|png      # 7 categories nested by numeric id

Splits come in three image variants, encoded as a directory suffix:
    raw   -> "<split>"                       (.jpg)
    enhwo -> "<split>_enhanced_wo_prompt"    (.png)
    enhw  -> "<split>_enhanced_w_prompt"     (.png)

OOD setup: among the 7 categories, the 5 with the highest train count are
in-distribution (label 0); the 2 lowest are out-of-distribution (label 1).
"""

import glob
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# category id -> name (from labels.txt)
CATEGORIES = {0: "bulk carrier", 1: "canoe", 2: "container ship", 3: "fishing boat",
              4: "liner", 5: "sailboat", 6: "warship"}
ID_CATS = [0, 1, 3, 5, 6]   # top-5 by train count (bulk carrier, canoe, fishing boat, sailboat, warship)
OOD_CATS = [2, 4]           # bottom-2 by train count (container ship, liner)

VARIANT_SUFFIX = {"raw": "", "enhwo": "_enhanced_wo_prompt", "enhw": "_enhanced_w_prompt"}


def split_dir(split: str, variant: str) -> str:
    """('train','enhwo') -> 'train_enhanced_wo_prompt'."""
    if variant not in VARIANT_SUFFIX:
        raise ValueError(f"unknown variant {variant}; expected one of {list(VARIANT_SUFFIX)}")
    return f"{split}{VARIANT_SUFFIX[variant]}"


def _cat_images(cat_dir: Path) -> list:
    return sorted(glob.glob(str(cat_dir / "*.jpg")) + glob.glob(str(cat_dir / "*.png")))


def list_split(root, split: str, variant: str):
    """Return (paths, cat_ids) for all 7 categories under root/<split_variant>/{0..6}, sorted per category."""
    base = Path(root) / split_dir(split, variant)
    if not base.exists():
        raise FileNotFoundError(f"split dir not found: {base}")
    paths, cats = [], []
    for c in range(7):
        imgs = _cat_images(base / str(c))
        paths.extend(imgs)
        cats.extend([c] * len(imgs))
        LOGGER.info(f"  {split}/{variant} cat{c} ({CATEGORIES[c]}): {len(imgs)} imgs")
    LOGGER.info(f"  {split}/{variant} total: {len(paths)} imgs")
    return paths, cats


def labels_from_cats(cat_ids, id_cats=ID_CATS) -> list:
    """0 if the category is in-distribution, 1 if out-of-distribution."""
    idset = set(id_cats)
    return [0 if c in idset else 1 for c in cat_ids]
