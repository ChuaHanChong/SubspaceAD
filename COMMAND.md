# Commands — Maritime OOD Detection

End-to-end pipeline for SubspaceAD on the maritime vessel datasets, across two backbones
(original RGB-pretrained + continual-IR) and three data conditions (clean / degraded / pooled).
Scripts live in `scripts/maritime/`; run from repo root (`/home/hcchua/SubspaceAD`).
Results land in `results_maritime/` (see its `README.md`); feature caches in `/data/hanchong/subspacead_cache/`.

## TL;DR — run everything

```bash
bash scripts/maritime/run_all.sh
```

`run_all.sh` builds the 3 missing caches (`ir_degraded`, `ir_continual`, `ir_degraded_continual`)
on GPU 0/1/2 while the A3 pooled grid runs on GPU 3, then drains all remaining grids through a
4-GPU work queue, then aggregates to `results_maritime/grid_summary_all.csv`. ~2–3 h cold.
Already-built caches are reused, so re-runs are grid-only (minutes).

---

## Shared configuration

| Item | Value |
|---|---|
| Python | `/data/hanchong/miniconda3/envs/subspacead/bin/python` |
| Arch | DINOv2 ViT-L/16 (24 layers, 1024-d), CLS token, 224 px |
| Normalization | mean `(0.5,0.5,0.5)`, std `(0.5,0.5,0.5)` |
| DINOv2 config | `…/Maritime-Vessel-Recognition/submodules/dinov2/dinov2/configs/train/vitl16_short.yaml` |
| DINOv2 submodule | `…/Maritime-Vessel-Recognition/submodules/dinov2` |
| Cache dir | `/data/hanchong/subspacead_cache/` (override with `CACHE_DIR=…`) |

### Backbones
| Tag | `--pretrained_weights` |
|---|---|
| orig | `…/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth` |
| continual | `…/artifacts-dinov2-all/pretraining/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth` |

### Dataset roots
| Dataset | `--data_root` |
|---|---|
| RGB (clean) | `/data/hanchong/images-splitted-3` |
| IR (clean) | `/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein` |
| IR (degraded) | `/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded` |

### Caches → experiments
| Cache | dataset / backbone | feeds |
|---|---|---|
| `rgb.npz` | RGB clean / orig | RGB baseline, A3 pool |
| `ir.npz` | IR clean / orig | IR baseline, A1 fit, A3 pool |
| `ir_degraded.npz` | IR degraded / orig | A1 eval, A2 |
| `rgbir_pool.npz` | RGB∪IR clean / orig | A3 |
| `ir_continual.npz` | IR clean / continual | B0, B1 fit |
| `ir_degraded_continual.npz` | IR degraded / continual | B1 eval, B2 |

---

## Step 1 — Feature extraction (`dump_chain.sh`)

Generic one-cache-per-call driver, parameterized by env vars: `DATA_ROOT`, `OUT_NAME`, `CKPT`
(default orig), `GPU` (default 0), `CHECK_DEGRADED` (1 = assert degraded root complete first).

```bash
# clean IR, continual backbone  → ir_continual.npz  (GPU 1)
DATA_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein \
CKPT=…/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth \
OUT_NAME=ir_continual.npz GPU=1 bash scripts/maritime/dump_chain.sh

# degraded IR, orig backbone    → ir_degraded.npz   (GPU 0, gated on degraded set being complete)
DATA_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded \
OUT_NAME=ir_degraded.npz GPU=0 CHECK_DEGRADED=1 bash scripts/maritime/dump_chain.sh

# degraded IR, continual backbone → ir_degraded_continual.npz  (GPU 2)
DATA_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded \
CKPT=…/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth \
OUT_NAME=ir_degraded_continual.npz GPU=2 CHECK_DEGRADED=1 bash scripts/maritime/dump_chain.sh
```

Each call runs the 5 fit sizes incrementally (100→10000) + val/test once, all 24 CLS layers.
The clean `rgb.npz` / `ir.npz` use the same driver with `CKPT`=orig on the clean roots —
`run_all.sh` builds all six caches (skip-if-present) automatically.

## Step 2 — Pool for fusion (`fuse_pool_features.py`)

```bash
python scripts/maritime/fuse_pool_features.py        # rgb.npz + ir.npz → rgbir_pool.npz (numpy merge, no GPU)
```

## Step 3 — Hyperparameter grid (`anomaly_detection.py`)

One experiment = 5 sizes × 1920 configs. `--eval_cache_file` makes fit and eval come from
different caches (cross-domain). Example, A1 (fit clean IR, eval degraded IR), size 5000, GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/maritime/anomaly_detection.py \
  --feature_cache_file /data/hanchong/subspacead_cache/ir.npz \
  --eval_cache_file    /data/hanchong/subspacead_cache/ir_degraded.npz \
  --data_root /data/hanchong/maritime-vessel-dataset-infrared-flux2-klein \
  --in_dist_subset In-distribution_5000perCat --size 5000 \
  --out_prefix grid_irX_5000_ \
  --results_root /home/hcchua/SubspaceAD/results_maritime/grids
```

Omit `--eval_cache_file` for same-cache experiments (A2, A3, B0, B2). `out_prefix` tags:
`grid_rgb_`, `grid_ir_`, `grid_irX_`, `grid_irdeg_`, `grid_rgbir_`, `grid_ircont_`, `grid_irXcont_`,
`grid_irdegcont_`. In practice use `run_all.sh` — it issues all of these across 4 GPUs.

## Step 4 — Aggregate (`aggregate_results.py`)

```bash
python scripts/maritime/aggregate_results.py \
  --results_root results_maritime/grids \
  --csv results_maritime/grid_summary_all.csv
```

Prints val-best per experiment + per-axis ablation slices; writes the master CSV. Default
`--results_root` is already `results_maritime/grids`.

## Step 5 — Balanced-test metrics (`eval_balanced.py`, optional)

```bash
python scripts/maritime/eval_balanced.py             # 400 ID + 400 OOD × 20 seeds, val-best configs
```

---

## Notes

- **GPUs**: 4× A100-40GB. `run_all.sh` uses all four (extractions on 0/1/2, A3 on 3; then a 4-GPU queue).
- **Outputs**: per-config `results_maritime/grids/grid_<exp>_<size>_<config>/metrics.json`; logs
  `results_maritime/grids/grid_<exp>_<size>_stdout.log`; summaries `results_maritime/*.csv`.
- **Resumable**: extraction is incremental (cache append); grids are idempotent (re-write per config).
- **Disk**: caches total ~113 GB on `/data` (keep them off `/`). Re-extraction is the only expensive step.
