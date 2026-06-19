# Commands — Maritime / ShipSpotting + Infiray OOD Detection

SubspaceAD across two SSL checkpoints (Base: RGB+IR · Continual: IR+degradation), the ShipSpotting
(maritime vessel) dataset (original / degraded / pooled), the Infiray IR dataset (in-domain + cross-domain),
and off-the-shelf HF DINOv2 (ViT-S/B/L). Scripts live in `scripts/maritime/`; run from the repo root
(`/home/hcchua/SubspaceAD`). Results land in `results_maritime/` (see its `README.md`); feature caches in
`/data/hanchong/subspacead_cache/`.

## TL;DR — run everything

```bash
bash scripts/maritime/run.sh all
```

`run.sh` chains the five stages **extract → grid → crossdomain → aggregate → speed**, each across a
4-GPU work queue. Caches/grids are skip-if-present, so warm re-runs are cheap. Run one stage, or scope
a stage to one family:

```bash
bash scripts/maritime/run.sh grid                 # just the grid stage (all families)
bash scripts/maritime/run.sh grid --only infiray  # just the Infiray grids
DRY=1 bash scripts/maritime/run.sh all | grep '^DRYCMD: '   # print every command, run nothing
```

## Architecture (per-stage + manifest)

| File | Role |
|---|---|
| `run.sh` | top driver: `run.sh [all\|extract\|grid\|crossdomain\|aggregate\|speed] [--only FAMILY]` |
| `lib.sh` | sourced: shared env (paths, checkpoints, `SIZES`) + the flock 4-GPU `run_queue` + `DRY=1` support |
| `manifest.sh` | sourced: the experiment tables — `CACHES`, `GRIDS`, `CROSS` — that every stage loops over |
| `extract.sh` | build feature caches (`--only local\|fuse\|infiray\|hf`) |
| `grid.sh` | PCA hyperparameter grids via `anomaly_detection.py` (`--only maritime\|rgbirX\|infiray\|hf`) |
| `crossdomain.sh` | `cross_grid.py` per detector + `infcross_report.py` (Variant A/B) |
| `aggregate.sh` | `aggregate.py` (CSV reports) + `eval_balanced.py` |
| `speed.sh` | `bench_speed.py` |

Every stage honors `DRY=1` (echoes `DRYCMD: <cmd>` instead of running) and `--only FAMILY`.

---

## Shared configuration

| Item | Value |
|---|---|
| Python | `/data/hanchong/miniconda3/envs/subspacead/bin/python` |
| Arch | DINOv2 ViT-L/16 (24 layers, 1024-d), CLS token, 224 px |
| Normalization | mean `(0.5,0.5,0.5)`, std `(0.5,0.5,0.5)` |
| DINOv2 config / submodule | `…/Maritime-Vessel-Recognition/submodules/dinov2/dinov2/configs/train/vitl16_short.yaml` / `…/submodules/dinov2` |
| Cache dir | `/data/hanchong/subspacead_cache/` |

### Backbones
| Tag | checkpoint |
|---|---|
| orig (Base) | `…/artifacts-dinov2-all/pretraining/ViT-L-16/eval/training_2348399/teacher_checkpoint.pth` |
| continual | `…/artifacts-dinov2-all/pretraining/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth` |

### Dataset roots
| Dataset | root |
|---|---|
| RGB (original) | `/data/hanchong/images-splitted-3` |
| IR (original) | `/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein` |
| IR (degraded) | `/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded` |
| Infiray | `/data/hanchong/other-infrared-datasets/processed-data/红外船舶数据库` (val=root A) + `…_Reversed` (test=root B) |

### Caches (all defined in `manifest.sh::CACHES`, built by `extract.sh`)
| Cache | builder | content |
|---|---|---|
| `rgb.npz`, `ir.npz`, `ir_degraded.npz`, `ir_continual.npz`, `ir_degraded_continual.npz` | `local` (`dump_chain.sh`) | ShipSpotting single-modality, 5 fit sizes + val/test |
| `rgbir_pool.npz` | `fuse` | RGB∪IR original (sample-pooled) |
| `rgbir_pool_origRGB_degIR.npz` | `fuse` | RGB-original ∪ IR-degraded |
| `infiray_{base,continual}_{raw,enhwo,enhw}.npz` | `infiray` (`extract_infiray.py`) | Infiray fit(train)/val(root A)/test(root B) |
| `hf_{s,b,l}_degraded.npz` | `hf` (`extract_features_hf.py`) | off-the-shelf DINOv2 on degraded IR |

---

## Stages

```bash
bash scripts/maritime/run.sh extract       # build any missing caches (local→dump_chain, fuse, infiray, hf)
bash scripts/maritime/run.sh grid          # 9 ShipSpotting + 18 Infiray + 3 HF grids (anomaly_detection.py)
bash scripts/maritime/run.sh crossdomain   # cross_grid.py × 8 detectors → infcross_report.py (Variant A/B)
bash scripts/maritime/run.sh aggregate     # aggregate.py CSVs + eval_balanced.py (balanced_test.csv)
bash scripts/maritime/run.sh speed         # bench_speed.py (speed_bench.csv)
```

Outputs (all under `results_maritime/`): per-config `grids/grid_<exp>_<size>_<cfg>/metrics.json`;
summary CSVs in `tables/` — `grid_summary_all.csv`, `best_per_experiment.csv`, `balanced_test.csv`,
`infiray_grid_best.csv`, `infcross_{maritimecfg,valselect}.csv`, `hf_grid_best.csv`, `speed_bench.csv`.

### Running the underlying scripts directly (for one-offs)

```bash
# one local cache (env-var driven; extract.sh calls this for the 5 ShipSpotting caches)
DATA_ROOT=/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein \
  CKPT=…/ViT-L-16-Continual-IR/eval/training_51199/teacher_checkpoint.pth \
  OUT_NAME=ir_continual.npz GPU=1 bash scripts/maritime/dump_chain.sh        # +CHECK_DEGRADED=1 for degraded roots

python scripts/maritime/fuse_pool_features.py                                # rgb.npz + ir.npz → rgbir_pool.npz

# one PCA grid (5 axes × sizes); --eval_cache_file for cross-domain (e.g. orig→degraded)
CUDA_VISIBLE_DEVICES=0 python scripts/maritime/anomaly_detection.py \
  --feature_cache_file …/ir.npz --eval_cache_file …/ir_degraded.npz \
  --data_root …/maritime-vessel-dataset-infrared-flux2-klein \
  --in_dist_subset In-distribution_5000perCat --size 5000 \
  --ood_subset Out-of-distribution --out_prefix grid_irX_5000_ \
  --results_root results_maritime/grids

# aggregate (merged tool): val-best per group, or --all (every config), or --detail (axis slices)
python scripts/maritime/aggregate.py --pattern 'grid_*' --all --csv results_maritime/tables/grid_summary_all.csv
python scripts/maritime/aggregate.py --pattern 'grid_infiray_*' --csv results_maritime/tables/infiray_grid_best.csv
python scripts/maritime/aggregate.py --pattern 'grid_ir_*' --detail        # per-axis report to stdout

python scripts/maritime/eval_balanced.py     # 400+400 × 20-seed balanced metrics → tables/balanced_test.csv
```

Shared Python primitives (`make_layer_configs`, `aggregate_features`, `compute_metrics`, `_chunked`,
`extract_per_layer_cls`, `balanced_metrics`, …) live in `scripts/maritime/_common.py`.

---

## Notes

- **GPUs**: 4× A100-40GB. Every stage drains its work through the shared 4-GPU `run_queue` in `lib.sh`.
  Run one stage at a time (don't oversubscribe GPUs across stages).
- **Resumable**: extraction is incremental (cache append) + skip-if-present; grids are idempotent (re-write per config).
- **Disk**: ShipSpotting caches ~17 GB each (pooled ~33 GB); keep them on `/data`. Re-extraction is the only expensive step.
- **Dry run**: `DRY=1 bash scripts/maritime/run.sh <stage>` prints the exact command set without executing — useful to preview or diff.
