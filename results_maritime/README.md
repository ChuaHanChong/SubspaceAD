# results_maritime — consolidated artifacts & results

All experiment outputs for the maritime OOD study live here.

## Layout
```
results_maritime/
├── REPORT.md, README.md, ANOMALY_CLASS.md   # committed deliverables (numbers live in REPORT.md)
├── tables/    # all summary CSVs            (git-ignored — regenerable)
├── grids/     # raw per-config metrics.json + per-grid logs + infcross_*_full.csv  (git-ignored)
└── logs/      # driver run logs (run_*.stdout.log)                                  (git-ignored)
```

## Documents
| File | Contents |
|---|---|
| `ANOMALY_CLASS.md` | **Method only** — the pixel→image adaptation, pipeline, hyperparameter axes, file inventory. No numbers. |
| `REPORT.md` | **All results**, dataset-first. **Part 1 — ShipSpotting**: Base & Continual (Summary · Balanced · Size sweep · Ablations), Base-vs-Continual + RGB∪IR fusion, off-the-shelf ViT-S/B/L. **Part 2 — Infiray** (raw): Base & Continual, cross-domain transfer + in-domain ablations. |
| `../COMMAND.md` | **Run commands** for every step. |

Each document owns one concern: method in `ANOMALY_CLASS.md`, results in `REPORT.md`, commands in `COMMAND.md` — no result is duplicated across documents.

## Data tables — `tables/`
> All `tables/*.csv` are **git-ignored** (local intermediates, regenerable); `REPORT.md` is the
> committed deliverable that embeds every headline number.

**Maritime study (8 experiments):**
| File | Contents |
|---|---|
| `grid_summary_all.csv` | Every config across all 8 experiments (66,709 rows): dataset, size, layer, agg, ev, score, drop_k, val/test AUROC/AUPR/FPR95, pca_k. |
| `best_per_experiment.csv` | Compact headline: the val-best config + test metrics for each of the 8 experiments. |
| `balanced_test.csv` | Class-balanced test metrics (400 ID + 400 OOD, 20-seed) per experiment: imbalanced vs balanced AUROC/AUPR, accuracy, OOD precision/recall/F1. |

**Extensions (REPORT Part 1 off-the-shelf + Part 2 Infiray):**
| File | Contents |
|---|---|
| `infiray_grid_best.csv` | Ask-3: val-best config + test metrics for each of the 18 Infiray fit×eval×backbone groups. |
| `infcross_maritimecfg.csv` | Ask-2 **Variant A**: each maritime detector's val-best config applied to Infiray (val+test), 21 rows. |
| `infcross_valselect.csv` | Ask-2 **Variant B**: best config re-selected on Infiray val → Infiray test, 21 rows. |
| `hf_grid_best.csv` | HF ViT-S/B/L deg→deg: val-best config + test metrics per (model, fit size). |
| `speed_bench.csv` | ViT-S/B/L compute, split by stage: throughput (img/s, b16) & latency (ms, b1) for feature-extraction / SubspaceAD-scoring / total, + offline PCA-fit time. |

## Raw per-config results — `grids/`
`grids/grid_<experiment>_<size>_<config>/metrics.json` — one dir per config (66,709 total).
Per-run logs: `grids/grid_<experiment>_<size>_stdout.log`. Experiment prefixes:

| Prefix | Model | Fit → Eval |
|---|---|---|
| `grid_rgb_*` | orig | RGB original→original |
| `grid_ir_*` | orig | IR original→original |
| `grid_irX_*` | orig | IR original→degraded |
| `grid_irdeg_*` | orig | IR degraded→degraded |
| `grid_rgbir_*` | orig | RGB∪IR pooled original |
| `grid_ircont_*` | continual | IR original→original |
| `grid_irXcont_*` | continual | IR original→degraded |
| `grid_irdegcont_*` | continual | IR degraded→degraded |
| `grid_infiray_{base,continual}_fit-{v}_eval-{v}_*` | base/continual | Ask-3: Infiray fit→eval per variant |
| `grid_hf_{s,b,l}_{size}_*` | HF ViT-S/B/L | degraded→degraded per fit size |
| `infcross_{exp}_full.csv` | maritime fit | Ask-2: full cross-domain grid per detector (all configs, all variants) |

## Scripts (in `scripts/maritime/`)
Per-stage drivers chained by `run.sh [all|extract|grid|crossdomain|aggregate|speed] [--only FAMILY]`
(`extract.sh` / `grid.sh` / `crossdomain.sh` / `aggregate.sh` / `speed.sh`), sourcing `lib.sh`
(shared env + 4-GPU `run_queue` + `DRY=1`) and `manifest.sh` (the `CACHES`/`GRIDS`/`CROSS` tables).
Workers: `dump_chain.sh` + `extract_features.py` (local caches), `extract_features_hf.py`,
`extract_infiray.py`, `fuse_pool_features.py`; `anomaly_detection.py` (grid); `cross_grid.py` +
`infcross_report.py` (cross-domain); `aggregate.py` (merged val-best / `--all` / `--detail`);
`eval_balanced.py`; `bench_speed.py`. Shared Python primitives live in `_common.py`. Loader:
`src/subspacead/data/infiray.py`.

## Feature caches (not here — too large)
`/data/hanchong/subspacead_cache/*.npz`: ShipSpotting `rgb`, `ir`, `ir_degraded`, `ir_continual`,
`ir_degraded_continual` (~17 GB each), `rgbir_pool` + `rgbir_pool_origRGB_degIR` (~33 GB); Infiray
`infiray_{base,continual}_{raw,enhwo,enhw}` (2.5 GB each); HF `hf_{s,b,l}_degraded`. Built by the
`extract` stage; consumed by `anomaly_detection.py` / `cross_grid.py`.

## Reproduce
Commands in `../COMMAND.md`. End-to-end: `bash scripts/maritime/run.sh all`.
