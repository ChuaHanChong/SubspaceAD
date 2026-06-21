# Commands — SubspaceAD OOD example

A single, general pipeline for image-level OOD detection with DINOv2 + PCA reconstruction-residual
scoring. A dataset is an **ID** (normal) image set and an **OOD** (anomaly) image set; the backbone
is either a local DINOv2 checkpoint or an off-the-shelf HF model.

## Quick start

Edit the variables at the top of `scripts/maritime/run.sh` (or pass them as env vars), then:

```bash
bash scripts/maritime/run.sh
```

It runs **extract features once → PCA hyperparameter grid → pick the val-best**.

### Inputs

| Variable | Meaning |
|---|---|
| `BACKBONE` | `local` (DINOv2 checkpoint) or `hf` (off-the-shelf) |
| `CKPT` / `DINO_CONFIG` / `DINO_SUBMOD` | local backbone: checkpoint + DINOv2 config + source tree |
| `HF_MODEL` | hf backbone, e.g. `facebook/dinov2-small` |
| `ID_FIT` | ID images to fit the PCA subspace |
| `ID_VAL` / `OOD_VAL` | validation negatives / positives (config selection) |
| `ID_TEST` / `OOD_TEST` | test negatives / positives (final report) |
| `NAME` / `CACHE` / `GRIDS` / `TABLES` | output prefix + directories |

### Examples (same pipeline, different inputs)

```bash
# ShipSpotting (local checkpoint)
NAME=shipspotting BACKBONE=local \
  CKPT=/my/ckpt.pth DINO_CONFIG=/my/dinov2/config.yaml DINO_SUBMOD=/my/dinov2 \
  ID_FIT=/data/ship/id/fit ID_VAL=/data/ship/id/val ID_TEST=/data/ship/id/test \
  OOD_VAL=/data/ship/ood/val OOD_TEST=/data/ship/ood/test \
  bash scripts/maritime/run.sh

# Infiray (same checkpoint, different folders)
NAME=infiray BACKBONE=local CKPT=/my/ckpt.pth DINO_CONFIG=... DINO_SUBMOD=... \
  ID_FIT=/data/infiray/id/fit ID_VAL=... ID_TEST=... OOD_VAL=... OOD_TEST=... \
  bash scripts/maritime/run.sh

# Off-the-shelf HF DINOv2 (no checkpoint)
NAME=hf BACKBONE=hf HF_MODEL=facebook/dinov2-small \
  ID_FIT=... ID_VAL=... ID_TEST=... OOD_VAL=... OOD_TEST=... \
  bash scripts/maritime/run.sh
```

## Files (`scripts/maritime/`)

| File | Role |
|---|---|
| `run.sh` | the whole flow; all paths are editable variables at the top |
| `extract_features.py` | DINOv2 CLS extraction — `--backbone local\|hf`, explicit ID/OOD folders → feature cache |
| `anomaly_detection.py` | PCA + scoring grid (fits on the cache's id_fit features) |
| `aggregate.py` | pick the val-best config per run → CSV |
| `_common.py` | shared primitives (layer configs, metrics, feature aggregation, image lister, extraction loop) |

## Hyperparameter grid

`anomaly_detection.py` sweeps the axes — layers, aggregation (mean/concat), PCA energy (`EV`),
score (reconstruction / mahalanobis / cosine / euclidean), `drop_k`. `aggregate.py` selects the best
**validation** AUROC and reports its **test** metrics (val-derived threshold → no leakage).
`--eval_cache_file` fits on one cache and evaluates on another (cross-domain).

## Run a step directly

```bash
python scripts/maritime/extract_features.py --backbone hf --hf_model facebook/dinov2-small \
  --id_fit ID/fit --id_val ID/val --id_test ID/test --ood_val OOD/val --ood_test OOD/test \
  --out_cache_file cache/example.npz
python scripts/maritime/anomaly_detection.py --feature_cache_file cache/example.npz \
  --out_prefix grid_example_ --results_root results/grids
python scripts/maritime/aggregate.py --pattern 'grid_example_*' --results_root results/grids \
  --csv results/tables/example_best.csv
```
