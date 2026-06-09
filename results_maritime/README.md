# results_maritime — consolidated artifacts & results

All experiment outputs for the maritime OOD study live here.

## Documents
| File | Contents |
|---|---|
| `ANOMALY_CLASS.md` | **Method only** — the pixel→image adaptation, pipeline, hyperparameter axes, file inventory. No numbers. |
| `REPORT.md` | **All results** — original baselines + ablations + threshold/balanced analysis, degradation robustness, continual backbone, RGB∪IR fusion. |
| `../COMMAND.md` | **Run commands** for every step. |

Each document owns one concern: method in `ANOMALY_CLASS.md`, results in `REPORT.md`, commands in `COMMAND.md` — no result is duplicated across documents.

## Data tables
| File | Contents |
|---|---|
| `grid_summary_all.csv` | Every config across all 8 experiments (66,709 rows): dataset, size, layer, agg, ev, score, drop_k, val/test AUROC/AUPR/FPR95, pca_k. |
| `best_per_experiment.csv` | Compact headline: the val-best config + test metrics for each of the 8 experiments. |
| `balanced_test.csv` | Class-balanced test metrics (400 ID + 400 OOD, 20-seed) per experiment: imbalanced vs balanced AUROC/AUPR, accuracy, OOD precision/recall/F1. |

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

## Feature caches (not here — too large)
`/data/hanchong/subspacead_cache/*.npz` (~17 GB each): `rgb`, `ir`, `ir_degraded`,
`ir_continual`, `ir_degraded_continual`, `rgbir_pool` (33 GB). Built by
`scripts/maritime/dump_chain.sh` / `fuse_pool_features.py`; consumed by `anomaly_detection.py`.

## Reproduce
Commands in `../COMMAND.md`. End-to-end: `scripts/maritime/run_all.sh`.
