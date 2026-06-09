"""Maritime Vessel Dataset adapter for SubspaceAD (image-level OOD detector).

Layout (under data_root):
    In-distribution_<subset>/train/C{id}/*.jpg   # per-class, used for fit
    In-distribution/test/*.jpg                   # flat,   used for ID test
    In-distribution/val/C{id}/*.jpg              # nested, used for ID val
    Out-of-distribution/test/*.jpg               # flat,   used for OOD test
    Out-of-distribution/val/C{id}/*.jpg          # nested, used for OOD val

Recursive glob in `list_test_paths` handles flat + nested uniformly. No
ground-truth pixel masks; labels are 0 (in-dist) or 1 (OOD) per image.
"""

import glob
import logging
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _list_class_folders(train_root: Path) -> list[Path]:
    return sorted(p for p in train_root.iterdir() if p.is_dir() and p.name.startswith("C"))


def list_fit_paths(
    in_dist_root: Path,
    max_per_class: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Sample fit-set ("normal") paths from the in-distribution train split.

    If max_per_class is None, returns every image. Sampling is deterministic per seed.
    """
    train_root = in_dist_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"train root not found: {train_root}")

    rng = random.Random(seed)
    paths: list[str] = []
    for cls_dir in _list_class_folders(train_root):
        cls_paths = sorted(glob.glob(str(cls_dir / "*.jpg")))
        if max_per_class is not None and len(cls_paths) > max_per_class:
            cls_paths = rng.sample(cls_paths, max_per_class)
        paths.extend(cls_paths)
        LOGGER.info(f"  fit: {cls_dir.name} -> {len(cls_paths)} imgs")
    LOGGER.info(f"  fit total: {len(paths)} imgs across {len(_list_class_folders(train_root))} classes")
    return paths


def list_test_paths(
    in_dist_root: Path,
    ood_root: Path,
    split: str = "test",
    max_id: int | None = None,
    max_ood: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Build a labeled test set: in-distribution = 0, OOD = 1.

    `split` is the leaf folder name ('test' or 'val').
    """
    rng = random.Random(seed)

    # Recursive: handles both flat (test/*.jpg) and nested (val/C{id}/*.jpg) layouts.
    id_glob = sorted(str(p) for p in (in_dist_root / split).rglob("*.jpg"))
    ood_glob = sorted(str(p) for p in (ood_root / split).rglob("*.jpg"))
    if not id_glob:
        raise FileNotFoundError(f"no in-dist {split} images at {in_dist_root / split}")
    if not ood_glob:
        raise FileNotFoundError(f"no OOD {split} images at {ood_root / split}")

    if max_id is not None and len(id_glob) > max_id:
        id_glob = rng.sample(id_glob, max_id)
    if max_ood is not None and len(ood_glob) > max_ood:
        ood_glob = rng.sample(ood_glob, max_ood)

    paths = id_glob + ood_glob
    labels = [0] * len(id_glob) + [1] * len(ood_glob)
    LOGGER.info(f"  eval: {len(id_glob)} in-dist + {len(ood_glob)} OOD = {len(paths)} imgs")
    return paths, labels


def resolve_roots(
    data_root: Path,
    in_dist_subset: str,
    ood_subset: str,
) -> tuple[Path, Path]:
    """Returns (in_dist_root, ood_root) under data_root."""
    data_root = Path(data_root)
    in_dist_root = data_root / in_dist_subset
    ood_root = data_root / ood_subset
    for r in (in_dist_root, ood_root):
        if not r.exists():
            raise FileNotFoundError(f"missing dataset root: {r}")
    return in_dist_root, ood_root
