# SubspaceAD OOD Detection — Results (ShipSpotting + Infiray)

Image-level OOD detection across two IR ship datasets, organized **dataset-first**.
**Part 1 — ShipSpotting** (maritime vessel dataset): the Base and Continual ViT-L backbones, each with
Summary · Balanced-test · Size sweep · Ablations, plus Base-vs-Continual and off-the-shelf backbones.
**Part 2 — Infiray**: a second IR ship dataset (real variant), Base and Continual, with cross-domain
transfer and in-domain ablations. Both datasets share the same two SSL backbones.
Method is in `ANOMALY_CLASS.md`; run commands in `../COMMAND.md`; data tables in `tables/`.

## Overview

Eval setup (both backbones): CLS token, 224 px, normalization `(0.5,0.5,0.5)`. Selection on **val AUROC**;
test reported on the val-selected config. Degraded data = Real-ESRGAN USM degradation of the IR set.

Both are the **same ViT-L/16 architecture** — only the SSL checkpoint differs:

| Variant | Checkpoint | SSL pretraining |
|---|---|---|
| **Base** | `ViT-L-16/eval/training_2348399` | self-supervised on **original RGB + synthetic IR** |
| **Continual** | `ViT-L-16-Continual-IR/eval/training_51199` | **initialized from Base**, then continued SSL on the **same synthetic IR with ESRGAN degradation mixed in** (`esrgan_prob=0.8`) |

Note: the IR used for SSL is **synthetic** (generated from RGB) for *both* backbones — Base uses the
clean synthetic IR, Continual continues on that same synthetic IR with ESRGAN degradation added.

Eval splits (per condition): full test = 1700 ID + 400 OOD; balanced test = 400 ID + 400 OOD (20-seed,
val-derived threshold held fixed). "original→degraded" = PCA fit on original images, evaluated on degraded;
"degraded→degraded" = fit and evaluated on degraded. Pooled = RGB and IR treated as separate samples
under one PCA. All result CSVs live in **`tables/`**: ShipSpotting — `grid_summary_all.csv` (66,709
configs), `best_per_experiment.csv`, `balanced_test.csv`; extensions — `infiray_grid_best.csv`,
`infcross_maritimecfg.csv` + `infcross_valselect.csv` (cross-domain Variant A/B), `hf_grid_best.csv`,
`speed_bench.csv`.

## How the OOD-detection subspace is derived (D → pca_k → k_eff)

Three steps turn the hyperparameters (`layers, agg, EV, drop_k`) into the number of components
actually used to score an image:

```
1.  D      =  1024                 if agg = mean
           =  n_layers × 1024      if agg = concat
              # n_layers: L1=1, L2=2, L4=4, L6=6, L8=8, L12=12, L18=18, Lmid=7

2.  pca_k  =  min { k : ( Σ_{i=1..k} λ_i ) / ( Σ_{i=1..D} λ_i ) ≥ EV }
              # λ = eigenvalues of the fit-feature covariance (desc); capped at min(D, N_fit−1)

3.  k_eff  =  pca_k − drop_k
              # components actually used to reconstruct “normal” → the OOD-scoring subspace
```

The anomaly score is the **residual energy in the complementary directions** — the dropped head
(`drop_k` components) plus the discarded tail (`D − pca_k` components). So `pca_k` and `k_eff` are
*consequences* of the hyperparameters, not knobs. (`core/pca.py`: `k = searchsorted(cumvar, EV)+1`;
`post_process/scoring.py` zeroes the top `drop_k` projections before reconstructing.)

**Worked example** — ShipSpotting RGB · original→original (layers=**L2**, agg=**concat**, EV=0.50, drop_k=20):
`D = 2 × 1024 = 2048`. PCA eigenvalues have no dominant direction (top = 1.57% of variance); walking
the cumulative ratio `cumvar(127)=0.4998 (<0.50) → cumvar(128)=0.5021 (≥0.50)` gives **pca_k = 128**.
Dropping the top 20 leaves **k_eff = 108** components reconstructing “normal”; the score is the residual
energy in the other `2048 − 108 = 1940` directions (top-20 nuisance head + 1920-dim tail).

The split across backbones is stark: the **base** winners truncate hard (k_eff ≈ 80–110 of 2048),
isolating a narrow mid-spectrum band; the **continual** winners use k_eff ≈ 950 of 6144 (drop_k=0) —
their IR-adapted features spread discriminative variance across a much higher-rank subspace.


## Part 1 — ShipSpotting dataset

### Setup

The ShipSpotting dataset (maritime vessel imagery) — 17 in-distribution ship classes + 4 held-out OOD
classes, in RGB and IR (original and Real-ESRGAN-degraded). Both backbones are fit and evaluated on this
dataset (in-domain). Each backbone gets the same 4-table treatment: Summary · Balanced-test detail ·
Size sweep · Ablations.

### Base backbone

#### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| RGB · original→original | 10000 | L2 | concat | 0.5 | reconstruction | 20 | 0.8664 | 0.8773 | 0.5440 | 0.8793±0.0087 | 0.8347±0.0159 |
| IR · original→original | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8340 | 0.8217 | 0.4259 | 0.8276±0.0101 | 0.7671±0.0156 |
| RGB+IR pooled · original→original (combined, 3400+800) | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8544 | 0.8547 | 0.5021 | 0.8558±0.0066 | 0.8042±0.0107 |
| ↳ RGB-only subset (1700+400) | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8673 | 0.8669 | 0.5447 | 0.8707±0.0083 | 0.8350±0.0148 |
| ↳ IR-only subset (1700+400) | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8407 | 0.8425 | 0.4620 | 0.8470±0.0099 | 0.7899±0.0166 |
| IR · original→degraded | 10000 | L8 | concat | 0.7 | reconstruction | 50 | 0.7155 | 0.7104 | 0.3175 | 0.7129±0.0141 | 0.6664±0.0165 |
| IR · degraded→degraded | 1000 | L4 | mean | 0.99 | reconstruction | 5 | 0.5932 | 0.5878 | 0.2388 | 0.5882±0.0104 | 0.5676±0.0107 |
| RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR) (combined, 3400+800) | 5000 | L2 | concat | 0.5 | reconstruction | 50 | 0.7253 | 0.7146 | 0.3822 | 0.7152±0.0070 | 0.7066±0.0086 |
| ↳ RGB-only subset (original, 1700+400) | 5000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8310 | 0.8252 | 0.4896 | 0.8291±0.0070 | 0.8001±0.0120 |
| ↳ IR-only subset (degraded, 1700+400) | 5000 | L2 | concat | 0.5 | reconstruction | 50 | 0.6334 | 0.6144 | 0.2470 | 0.6162±0.0134 | 0.5858±0.0149 |

The pooled detector is one PCA over RGB∪IR samples; the two ↳ rows just re-score the same detector on
each modality's slice of the test set (balanced metrics are per-slice too).

#### Balanced-test detail (400 ID + 400 OOD, 20-seed mean)

| Experiment | accuracy ↑ | OOD precision ↑ | OOD recall ↑ | OOD F1 ↑ |
|:--|--:|--:|--:|--:|
| RGB · original→original | 0.7904 | 0.7983 | 0.7775 | 0.7877 |
| IR · original→original | 0.7583 | 0.7204 | 0.8450 | 0.7777 |
| RGB+IR pooled · original→original (combined) | 0.7787 | 0.7581 | 0.8187 | 0.7872 |
| ↳ RGB-only subset | 0.7943 | 0.7650 | 0.8500 | 0.8052 |
| ↳ IR-only subset | 0.7645 | 0.7532 | 0.7875 | 0.7699 |
| IR · original→degraded | 0.6558 | 0.6153 | 0.8325 | 0.7076 |
| IR · degraded→degraded | 0.5748 | 0.5844 | 0.5200 | 0.5502 |
| RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR) (combined) | 0.6423 | 0.6837 | 0.5300 | 0.5971 |
| ↳ RGB-only subset (original) | 0.7520 | 0.7191 | 0.8275 | 0.7695 |
| ↳ IR-only subset (degraded) | 0.5376 | 0.5972 | 0.2325 | 0.3346 |

#### Size sweep

| fit size | RGB · original→original | IR · original→original | IR · original→degraded | IR · degraded→degraded | RGB+IR pooled · original→original | RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR) |
|--:|--:|--:|--:|--:|--:|--:|
| 100 | 0.838 / 0.857 | 0.817 / 0.822 | 0.699 / 0.702 | 0.585 / 0.579 | 0.821 / 0.834 | 0.714 / 0.710 |
| 500 | 0.861 / 0.865 | 0.821 / 0.823 | 0.709 / 0.712 | 0.592 / 0.587 | 0.846 / 0.849 | 0.723 / 0.707 |
| 1000 | 0.853 / 0.860 | 0.828 / 0.820 | 0.709 / 0.708 | 0.593 / 0.588 | 0.846 / 0.846 | 0.720 / 0.710 |
| 5000 | 0.864 / 0.867 | 0.833 / 0.822 | 0.709 / 0.695 | 0.590 / 0.580 | 0.852 / 0.854 | 0.725 / 0.715 |
| 10000 | 0.866 / 0.877 | 0.834 / 0.822 | 0.715 / 0.710 | 0.591 / 0.576 | 0.854 / 0.855 | 0.725 / 0.715 |

_(val / test AUROC; val-best config at each fit size)_

#### Ablations

_Each table varies one axis; all other hyperparameters held at the experiment's val-best (listed per table). Cells = val / test AUROC._

**RGB · original→original**

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.5 · score=reconstruction · drop_k=20_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.811 | 0.866 | 0.815 | 0.790 | 0.802 | 0.788 | 0.750 | 0.561 |
| test AUROC | 0.837 | 0.877 | 0.826 | 0.803 | 0.810 | 0.806 | 0.776 | 0.566 |

_Vary **EV** · fixed: size=10000 · layers=L2 · agg=concat · score=reconstruction · drop_k=20_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.866 | 0.813 | 0.616 | 0.550 | 0.490 | 0.476 |
| test AUROC | 0.877 | 0.837 | 0.630 | 0.558 | 0.491 | 0.476 |

_Vary **Score** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · drop_k=20_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.866 | 0.570 | 0.712 | 0.559 |
| test AUROC | 0.877 | 0.556 | 0.731 | 0.542 |

_Vary **drop_k** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.803 | 0.843 | 0.866 | 0.862 | 0.836 |
| test AUROC | 0.816 | 0.852 | 0.877 | 0.865 | 0.841 |


**IR · original→original**

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.5 · score=reconstruction · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.733 | 0.834 | 0.763 | 0.731 | 0.760 | 0.736 | 0.674 | 0.527 |
| test AUROC | 0.715 | 0.822 | 0.773 | 0.751 | 0.766 | 0.746 | 0.683 | 0.521 |

_Vary **EV** · fixed: size=10000 · layers=L2 · agg=concat · score=reconstruction · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.834 | 0.772 | 0.582 | 0.526 | 0.488 | 0.480 |
| test AUROC | 0.822 | 0.754 | 0.555 | 0.502 | 0.464 | 0.457 |

_Vary **Score** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.834 | 0.545 | 0.687 | 0.547 |
| test AUROC | 0.822 | 0.569 | 0.659 | 0.569 |

_Vary **drop_k** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.765 | 0.777 | 0.803 | 0.834 | 0.811 |
| test AUROC | 0.776 | 0.784 | 0.808 | 0.822 | 0.810 |


**IR · original→degraded**

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.7 · score=reconstruction · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.645 | 0.704 | 0.699 | 0.705 | 0.715 | 0.714 | 0.648 | 0.540 |
| test AUROC | 0.626 | 0.683 | 0.696 | 0.707 | 0.710 | 0.698 | 0.635 | 0.534 |

_Vary **EV** · fixed: size=10000 · layers=L8 · agg=concat · score=reconstruction · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.694 | 0.715 | 0.621 | 0.566 | 0.516 | 0.506 |
| test AUROC | 0.688 | 0.710 | 0.608 | 0.545 | 0.491 | 0.482 |

_Vary **Score** · fixed: size=10000 · layers=L8 · agg=concat · EV=0.7 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.715 | 0.498 | 0.643 | 0.490 |
| test AUROC | 0.710 | 0.483 | 0.651 | 0.472 |

_Vary **drop_k** · fixed: size=10000 · layers=L8 · agg=concat · EV=0.7 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.676 | 0.687 | 0.698 | 0.715 | 0.690 |
| test AUROC | 0.685 | 0.698 | 0.705 | 0.710 | 0.680 |


**IR · degraded→degraded**

_Vary **Layer** · fixed: size=1000 · agg=mean · EV=0.99 · score=reconstruction · drop_k=5_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.576 | 0.589 | 0.593 | 0.587 | 0.551 | 0.488 | 0.518 | 0.468 |
| test AUROC | 0.560 | 0.575 | 0.588 | 0.583 | 0.549 | 0.490 | 0.515 | 0.480 |

_Vary **EV** · fixed: size=1000 · layers=L4 · agg=mean · score=reconstruction · drop_k=5_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.502 | 0.527 | 0.574 | 0.590 | 0.593 | 0.593 |
| test AUROC | 0.505 | 0.540 | 0.569 | 0.582 | 0.588 | 0.588 |

_Vary **Score** · fixed: size=1000 · layers=L4 · agg=mean · EV=0.99 · drop_k=5_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.593 | 0.460 | 0.577 | 0.475 |
| test AUROC | 0.588 | 0.454 | 0.572 | 0.470 |

_Vary **drop_k** · fixed: size=1000 · layers=L4 · agg=mean · EV=0.99 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.478 | 0.593 | 0.521 | 0.572 | 0.557 |
| test AUROC | 0.479 | 0.588 | 0.533 | 0.587 | 0.580 |


**RGB+IR pooled · original→original**

_Combined val/test, each split by modality into ↳ RGB-only / ↳ IR-only; same pooled detector._

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.5 · score=reconstruction · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.766 | 0.854 | 0.784 | 0.754 | 0.779 | 0.767 | 0.707 | 0.557 |
| ↳ RGB-only | 0.786 | 0.867 | 0.804 | 0.775 | 0.798 | 0.788 | 0.732 | 0.562 |
| ↳ IR-only | 0.745 | 0.841 | 0.763 | 0.734 | 0.761 | 0.749 | 0.685 | 0.553 |
| test AUROC | 0.766 | 0.855 | 0.796 | 0.772 | 0.788 | 0.779 | 0.713 | 0.551 |
| ↳ RGB-only | 0.790 | 0.867 | 0.807 | 0.780 | 0.796 | 0.790 | 0.725 | 0.560 |
| ↳ IR-only | 0.742 | 0.843 | 0.785 | 0.766 | 0.783 | 0.773 | 0.705 | 0.546 |

_Vary **EV** · fixed: size=10000 · layers=L2 · agg=concat · score=reconstruction · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.854 | 0.787 | 0.580 | 0.522 | 0.480 | 0.472 |
| ↳ RGB-only | 0.867 | 0.788 | 0.580 | 0.523 | 0.480 | 0.471 |
| ↳ IR-only | 0.841 | 0.786 | 0.584 | 0.523 | 0.479 | 0.471 |
| test AUROC | 0.855 | 0.783 | 0.572 | 0.512 | 0.468 | 0.459 |
| ↳ RGB-only | 0.867 | 0.788 | 0.580 | 0.523 | 0.480 | 0.471 |
| ↳ IR-only | 0.843 | 0.778 | 0.564 | 0.502 | 0.457 | 0.449 |

_Vary **Score** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.854 | 0.547 | 0.712 | 0.539 |
| ↳ RGB-only | 0.867 | 0.575 | 0.703 | 0.565 |
| ↳ IR-only | 0.841 | 0.518 | 0.723 | 0.513 |
| test AUROC | 0.855 | 0.550 | 0.705 | 0.541 |
| ↳ RGB-only | 0.867 | 0.572 | 0.699 | 0.564 |
| ↳ IR-only | 0.843 | 0.527 | 0.713 | 0.517 |

_Vary **drop_k** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.788 | 0.809 | 0.836 | 0.854 | 0.824 |
| ↳ RGB-only | 0.801 | 0.823 | 0.855 | 0.867 | 0.842 |
| ↳ IR-only | 0.774 | 0.793 | 0.816 | 0.841 | 0.805 |
| test AUROC | 0.800 | 0.819 | 0.844 | 0.855 | 0.825 |
| ↳ RGB-only | 0.809 | 0.829 | 0.856 | 0.867 | 0.845 |
| ↳ IR-only | 0.791 | 0.808 | 0.832 | 0.843 | 0.804 |


**RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR)**

_Vary **Layer** · fixed: size=5000 · agg=concat · EV=0.5 · score=reconstruction · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.619 | 0.725 | 0.666 | 0.640 | 0.640 | 0.586 | 0.537 | 0.484 |
| test AUROC | 0.610 | 0.715 | 0.669 | 0.643 | 0.634 | 0.576 | 0.532 | 0.473 |

_Vary **EV** · fixed: size=5000 · layers=L2 · agg=concat · score=reconstruction · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.725 | 0.707 | 0.590 | 0.566 | 0.549 | 0.544 |
| test AUROC | 0.715 | 0.709 | 0.596 | 0.573 | 0.552 | 0.548 |

_Vary **Score** · fixed: size=5000 · layers=L2 · agg=concat · EV=0.5 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.725 | 0.509 | 0.544 | 0.509 |
| test AUROC | 0.715 | 0.499 | 0.547 | 0.500 |

_Vary **drop_k** · fixed: size=5000 · layers=L2 · agg=concat · EV=0.5 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.602 | 0.680 | 0.697 | 0.725 | 0.687 |
| test AUROC | 0.598 | 0.674 | 0.684 | 0.715 | 0.675 |


### Continual backbone

#### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→original | 5000 | L6 | concat | 0.99 | reconstruction | 0 | 0.8703 | 0.8814 | 0.5784 | 0.8820±0.0077 | 0.8458±0.0153 |
| IR · original→degraded | 1000 | L8 | concat | 0.7 | reconstruction | 5 | 0.8371 | 0.8358 | 0.5104 | 0.8358±0.0100 | 0.8067±0.0159 |
| IR · degraded→degraded | 5000 | L6 | concat | 0.99 | reconstruction | 0 | 0.8516 | 0.8600 | 0.5074 | 0.8608±0.0092 | 0.8091±0.0176 |

#### Balanced-test detail (400 ID + 400 OOD, 20-seed mean)

| Experiment | accuracy ↑ | OOD precision ↑ | OOD recall ↑ | OOD F1 ↑ |
|:--|--:|--:|--:|--:|
| IR · original→original | 0.8086 | 0.8364 | 0.7675 | 0.8005 |
| IR · original→degraded | 0.7425 | 0.7683 | 0.6950 | 0.7297 |
| IR · degraded→degraded | 0.7925 | 0.7637 | 0.8475 | 0.8034 |

#### Size sweep

| fit size | IR · original→original | IR · original→degraded | IR · degraded→degraded |
|--:|--:|--:|--:|
| 100 | 0.821 / 0.821 | 0.802 / 0.800 | 0.775 / 0.769 |
| 500 | 0.869 / 0.879 | 0.833 / 0.832 | 0.850 / 0.853 |
| 1000 | 0.869 / 0.879 | 0.837 / 0.836 | 0.851 / 0.857 |
| 5000 | 0.870 / 0.881 | 0.834 / 0.834 | 0.852 / 0.860 |
| 10000 | 0.868 / 0.882 | 0.834 / 0.834 | 0.851 / 0.860 |

_(val / test AUROC; val-best config at each fit size)_

#### Ablations

_Each table varies one axis; all other hyperparameters held at the experiment's val-best (listed per table). Cells = val / test AUROC._

**IR · original→original**

_Vary **Layer** · fixed: size=5000 · agg=concat · EV=0.99 · score=reconstruction · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.753 | 0.821 | 0.870 | 0.870 | 0.840 | 0.835 | 0.827 | 0.602 |
| test AUROC | 0.763 | 0.840 | 0.883 | 0.881 | 0.851 | 0.845 | 0.837 | 0.612 |

_Vary **EV** · fixed: size=5000 · layers=L6 · agg=concat · score=reconstruction · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.797 | 0.818 | 0.850 | 0.866 | 0.870 | 0.842 |
| test AUROC | 0.814 | 0.830 | 0.867 | 0.878 | 0.881 | 0.860 |

_Vary **Score** · fixed: size=5000 · layers=L6 · agg=concat · EV=0.99 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.870 | 0.845 | 0.860 | 0.793 |
| test AUROC | 0.881 | 0.860 | 0.869 | 0.806 |

_Vary **drop_k** · fixed: size=5000 · layers=L6 · agg=concat · EV=0.99 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.870 | 0.699 | 0.558 | 0.551 | 0.592 |
| test AUROC | 0.881 | 0.702 | 0.581 | 0.560 | 0.605 |


**IR · original→degraded**

_Vary **Layer** · fixed: size=1000 · agg=concat · EV=0.7 · score=reconstruction · drop_k=5 · L18 concat rank-deficient — omitted_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.562 | 0.709 | 0.787 | 0.817 | 0.837 | 0.748 | 0.532 |
| test AUROC | 0.541 | 0.696 | 0.790 | 0.824 | 0.836 | 0.756 | 0.556 |

_Vary **EV** · fixed: size=1000 · layers=L8 · agg=concat · score=reconstruction · drop_k=5_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.822 | 0.837 | 0.814 | 0.784 | 0.696 | 0.653 |
| test AUROC | 0.826 | 0.836 | 0.828 | 0.791 | 0.698 | 0.653 |

_Vary **Score** · fixed: size=1000 · layers=L8 · agg=concat · EV=0.7 · drop_k=5_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.837 | 0.630 | 0.728 | 0.574 |
| test AUROC | 0.836 | 0.641 | 0.714 | 0.589 |

_Vary **drop_k** · fixed: size=1000 · layers=L8 · agg=concat · EV=0.7 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.800 | 0.837 | 0.786 | 0.799 | 0.783 |
| test AUROC | 0.797 | 0.836 | 0.796 | 0.802 | 0.788 |


**IR · degraded→degraded**

_Vary **Layer** · fixed: size=5000 · agg=concat · EV=0.99 · score=reconstruction · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.706 | 0.782 | 0.847 | 0.852 | 0.816 | 0.801 | 0.786 | 0.571 |
| test AUROC | 0.714 | 0.777 | 0.852 | 0.860 | 0.828 | 0.817 | 0.800 | 0.566 |

_Vary **EV** · fixed: size=5000 · layers=L6 · agg=concat · score=reconstruction · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.760 | 0.784 | 0.820 | 0.837 | 0.852 | 0.821 |
| test AUROC | 0.766 | 0.784 | 0.829 | 0.842 | 0.860 | 0.825 |

_Vary **Score** · fixed: size=5000 · layers=L6 · agg=concat · EV=0.99 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.852 | 0.811 | 0.839 | 0.743 |
| test AUROC | 0.860 | 0.815 | 0.847 | 0.748 |

_Vary **drop_k** · fixed: size=5000 · layers=L6 · agg=concat · EV=0.99 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.852 | 0.656 | 0.527 | 0.521 | 0.539 |
| test AUROC | 0.860 | 0.666 | 0.564 | 0.550 | 0.567 |


### Analysis — Base vs Continual (IR robustness) & RGB∪IR fusion

The continual (ESRGAN-augmented) backbone barely degrades across conditions; the base collapses
toward chance on degraded data.

| IR scenario | Base test AUROC | Continual test AUROC | Δ |
|:--|--:|--:|--:|
| original → original | 0.8217 | **0.8814** | +0.060 |
| original → degraded | 0.7104 | **0.8358** | **+0.125** |
| degraded → degraded | 0.5878 | **0.8600** | **+0.272** |

Balanced AUROC tells the same story (0.828 / 0.713 / 0.588 vs 0.882 / 0.836 / 0.861).

**Fusion (RGB∪IR pooled, base backbone)**: test AUROC **0.8547** — above original IR (0.8217),
just below original RGB (0.8773). One PCA subspace serves both modalities without per-modality tuning.
The Base Summary splits this pooled eval into its RGB-only (0.8669) and IR-only (0.8425) subsets: pooling
**helps IR** (+0.021 vs the standalone IR detector's 0.8217) and **slightly costs RGB** (−0.010 vs
standalone 0.8773) — the shared subspace borrows RGB's cleaner structure to lift IR.

### Off-the-shelf backbones — DINOv2 ViT-S/B/L (degraded IR)

Original SubspaceAD (HF DINOv2, **identical PCA-fit + grid** as the two backbones above) on ShipSpotting
degraded IR (deg→deg), full size sweep + a compute-speed benchmark — the same treatment as a backbone.

#### Summary

| Model | params | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ |
|:--|--:|--:|:--|:--|--:|:--|--:|--:|--:|--:|
| ViT-S | 22.1 M | 500 | L2 | mean | 0.9 | euclidean | 5 | 0.6667 | 0.6280 | 0.3236 |
| ViT-B | 86.6 M | 100 | L1 | concat | 0.7 | mahalanobis | 20 | 0.6842 | 0.6029 | 0.3502 |
| ViT-L | 304.4 M | 100 | L1 | concat | 0.7 | euclidean | 20 | 0.6806 | 0.6619 | 0.4219 |

#### Size sweep

| fit size | ViT-S | ViT-B | ViT-L |
|--:|--:|--:|--:|
| 100 | 0.663 / 0.626 | 0.684 / 0.603 | 0.681 / 0.662 |
| 500 | 0.667 / 0.628 | 0.677 / 0.600 | 0.675 / 0.664 |
| 1000 | 0.666 / 0.628 | 0.673 / 0.605 | 0.673 / 0.659 |
| 5000 | 0.665 / 0.627 | 0.673 / 0.606 | 0.673 / 0.660 |
| 10000 | 0.666 / 0.627 | 0.674 / 0.605 | 0.673 / 0.660 |

_(val / test AUROC; val-best config at each fit size)_

#### Speed (same GPU, 224 px, FP32)

Per-image inference split into its two stages: **feature extraction** (DINOv2 forward → per-layer CLS →
host) and **SubspaceAD scoring** (PCA reconstruction residual, CPU — fixed reference config L2 concat ·
EV0.5 · rec · dk20). The offline PCA *fit* (~0.1 s, training cost) is not part of inference.

**Throughput — batch 16 (img/s ↑):**

| Model | params | embed / layers | feature extraction | SubspaceAD | total |
|:--|--:|:--|--:|--:|--:|
| ViT-S | 22.1 M | 384 / 12 | 216.5 | 32190.6 | 215.1 |
| ViT-B | 86.6 M | 768 / 12 | 156.6 | 9399.9 | 154.0 |
| ViT-L | 304.4 M | 1024 / 24 | 69.7 | 5791.7 | 68.9 |

**Latency — batch 1 (ms/img ↓):**

| Model | feature extraction | SubspaceAD | total |
|:--|--:|--:|--:|
| ViT-S | 11.69 | 0.10 | 11.79 |
| ViT-B | 12.56 | 1.24 | 13.80 |
| ViT-L | 25.38 | 1.93 | 27.31 |

SubspaceAD scoring is **<10% of total latency** (≈1% / 9% / 7% for S/B/L) and **<1% of the throughput
cost** — feature extraction dominates end to end. (Extraction here includes the GPU→host transfer of the
CLS features, so absolute throughput is below a forward-only measurement.)

#### Ablations

_Each table varies one axis; the others are held at the model's val-best **at the 10000/class fit**.
That fit is used (rather than the Summary's val-selected 100–500/class) only so every layer is
rank-sufficient — HF is size-invariant (see Size sweep), so the operating point is equivalent. Cells = val / test AUROC._

**ViT-S** (L2 · mean · EV0.9 · euc · dk5)

_Vary **Layer** · fixed: agg=mean · EV=0.9 · score=euclidean · drop_k=5_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.662 | 0.666 | 0.657 | 0.652 | 0.651 | 0.651 | 0.517 |
| test AUROC | 0.617 | 0.627 | 0.628 | 0.624 | 0.622 | 0.621 | 0.508 |

_Vary **EV** · fixed: layer=L2 · agg=mean · score=euclidean · drop_k=5_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.636 | 0.658 | 0.666 | 0.665 | 0.665 | 0.665 |
| test AUROC | 0.596 | 0.618 | 0.627 | 0.630 | 0.631 | 0.631 |

_Vary **Score** · fixed: layer=L2 · agg=mean · EV=0.9 · drop_k=5_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.472 | 0.651 | 0.463 | 0.666 |
| test AUROC | 0.469 | 0.616 | 0.457 | 0.627 |

_Vary **drop_k** · fixed: layer=L2 · agg=mean · EV=0.9 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.598 | 0.666 | 0.634 | 0.628 | 0.618 |
| test AUROC | 0.571 | 0.627 | 0.611 | 0.599 | 0.594 |

**ViT-B** (L1 · concat · EV0.99 · euc · dk20)

_Vary **Layer** · fixed: agg=concat · EV=0.99 · score=euclidean · drop_k=20_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.674 | 0.668 | 0.667 | 0.666 | 0.666 | 0.666 | 0.527 |
| test AUROC | 0.605 | 0.603 | 0.605 | 0.605 | 0.605 | 0.605 | 0.519 |

_Vary **EV** · fixed: layer=L1 · agg=concat · score=euclidean · drop_k=20_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.500 | 0.673 | 0.672 | 0.674 | 0.674 | 0.674 |
| test AUROC | 0.500 | 0.600 | 0.601 | 0.603 | 0.605 | 0.606 |

_Vary **Score** · fixed: layer=L1 · agg=concat · EV=0.99 · drop_k=20_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.489 | 0.662 | 0.443 | 0.674 |
| test AUROC | 0.477 | 0.605 | 0.451 | 0.605 |

_Vary **drop_k** · fixed: layer=L1 · agg=concat · EV=0.99 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.605 | 0.672 | 0.674 | 0.653 | 0.659 |
| test AUROC | 0.561 | 0.607 | 0.605 | 0.592 | 0.602 |

**ViT-L** (L1 · concat · EV0.5 · cos · dk0)

_Vary **Layer** · fixed: agg=concat · EV=0.5 · score=cosine · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.673 | 0.673 | 0.663 | 0.656 | 0.656 | 0.656 | 0.656 | 0.537 |
| test AUROC | 0.660 | 0.659 | 0.651 | 0.643 | 0.643 | 0.643 | 0.643 | 0.518 |

_Vary **EV** · fixed: layer=L1 · agg=concat · score=cosine · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.673 | 0.632 | 0.637 | 0.637 | 0.636 | 0.634 |
| test AUROC | 0.660 | 0.615 | 0.613 | 0.615 | 0.616 | 0.612 |

_Vary **Score** · fixed: layer=L1 · agg=concat · EV=0.5 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.651 | 0.444 | 0.673 | 0.330 |
| test AUROC | 0.631 | 0.464 | 0.660 | 0.344 |

_Vary **drop_k** · fixed: layer=L1 · agg=concat · EV=0.5 · score=cosine_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.673 | 0.564 | 0.539 | 0.539 | 0.539 |
| test AUROC | 0.660 | 0.532 | 0.523 | 0.523 | 0.523 |

#### Analysis — off-the-shelf vs Base & Continual (deg→deg)

| Backbone | pretrain | test AUROC ↑ |
|:--|:--|--:|
| HF ViT-S | ImageNet SSL | 0.628 |
| HF ViT-B | ImageNet SSL | 0.603 |
| HF ViT-L | ImageNet SSL | **0.662** |
| Base (ViT-L) | ShipSpotting RGB+IR SSL | 0.588 |
| Continual (ViT-L) | ShipSpotting IR + degradation SSL | **0.860** |

- **Off-the-shelf DINOv2 is mediocre on degraded IR (0.60–0.66).** HF ViT-L (0.662) tops the off-the-shelf
  set and *beats the ShipSpotting Base backbone* (0.588) — generic ImageNet-SSL features survive degradation
  better than ShipSpotting RGB+IR-SSL features that never saw degradation.
- **But all HF models fall far short of the Continual backbone (0.860).** Pretraining that *includes the
  degradation* dominates any amount of generic scale — domain-matched SSL ≫ off-the-shelf size.
- **Bigger isn't monotonic:** ViT-B (0.603) < ViT-S (0.628). Only ViT-L's extra depth/width helps.
- **Fit size barely matters** — all models plateau by 100 images/class.
- **Cost:** ViT-L is **~3× slower** end-to-end (68.9 vs 215.1 img/s, batch-16) and 14× the params of ViT-S
  for +0.03 AUROC. The **SubspaceAD scoring stage is negligible** — ≤2 ms/img and <10% of total latency
  (the PCA projection + reconstruction residual is a handful of CPU matmuls); **feature extraction dominates
  end to end.**

## Part 2 — Infiray dataset (real IR)

A second, independent IR ship dataset (**Infiray**, 7 categories) tests how the ShipSpotting backbones
and detectors generalize to a new IR domain. Only the **real** (raw-capture) image variant is used here;
the AI-`enhanced (no-prompt / prompt)` variants live in `tables/` and don't change the story.

### Setup

Shared: Infiray has 7 categories; **val = `红外船舶数据库`**, **test = `红外船舶数据库_Reversed`**.
The two experiment types differ in **where the PCA is fit** and **what counts as OOD** (reported in
this order):

- **Cross-domain** — PCA is **kept on a ShipSpotting fit** (no Infiray training); a ShipSpotting detector
  is applied to Infiray. Negatives = **Infiray, all 7 categories (ID)**; positives = that detector's
  matched **ShipSpotting OOD** = the held-out ShipSpotting ship classes it was tested against in Part 1.
  Detectors are shown only when their OOD is **degraded** (Real-ESRGAN) IR, which roughly matches Infiray's
  low resolution:
  - `IR · original→degraded`, `IR · degraded→degraded` → **degraded**-IR OOD
  - `RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR)` → **mixed** OOD (degraded IR + *original RGB*); included but
    flagged, since its RGB half is still sharp

  The pure-`original→original` detectors (`IR · original→original`, `RGB+IR pooled · original→original`) are
  **excluded**: their OOD is the *original* (sharp — and for pooled, also RGB) ShipSpotting set, so pairing
  it against low-resolution Infiray ID would let the detector separate by **resolution/modality** rather
  than ship semantics. (Continual-backbone detectors use the same OOD sets in the Continual feature space.)
  Two configs per detector: **Variant A** = its frozen ShipSpotting val-best config; **Variant B** = config
  re-selected on Infiray val. Each cross-domain AUROC asks *does this detector rank ShipSpotting-OOD ships
  above Infiray ships?* — so it still carries a dataset-domain confound (see the findings).
- **In-domain** — PCA is **fit on Infiray `train`**, with OOD *internal to Infiray*: **ID = the 5
  highest-count categories {0,1,3,5,6}, OOD = the 2 rarest {2,4}**. Selection on val, reporting on test.
  Same Summary + per-axis ablation treatment as a ShipSpotting backbone.

### Base backbone

#### Cross-domain — ShipSpotting Base detectors → Infiray (real)

PCA kept on each detector's ShipSpotting fit; negatives = Infiray (all 7 cats, ID), positives = that
detector's matched ShipSpotting OOD. Detectors whose OOD is **degraded** IR (Real-ESRGAN) are shown,
since that roughly matches Infiray's low resolution. The pure-original detectors (`IR · original→original`,
`RGB+IR pooled · original→original`) are excluded — their sharp OOD vs blurry Infiray would let the score
separate by resolution (see Setup). The mixed `RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR)` detector is
included but flagged: its OOD's **RGB half is still sharp**, so it partially carries the confound (Variant A
sits near chance rather than inverting). Balanced = equal ID/OOD subsample, 20-seed, val-threshold fixed.

**Variant A — frozen ShipSpotting val-best config**

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→degraded | 10000 | L8 | concat | 0.7 | reconstruction | 50 | 0.3163 | 0.3191 | 0.2650 | 0.3209±0.0103 | 0.3833±0.0043 |
| IR · degraded→degraded | 1000 | L4 | mean | 0.99 | reconstruction | 5 | 0.4424 | 0.4283 | 0.3051 | 0.4310±0.0103 | 0.4325±0.0062 |
| RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR) | 5000 | L2 | concat | 0.5 | reconstruction | 50 | 0.5820 | 0.5678 | 0.6618 | 0.5677±0.0068 | 0.6339±0.0060 |

**Variant B — config re-selected on Infiray val**

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→degraded | 10000 | Lmid | concat | 0.7 | euclidean | 50 | 0.7130 | 0.6916 | 0.5129 | 0.6911±0.0072 | 0.6484±0.0082 |
| IR · degraded→degraded | 1000 | L1 | mean | 0.5 | euclidean | 20 | 0.6475 | 0.6487 | 0.5350 | 0.6459±0.0086 | 0.6577±0.0093 |
| RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR) | 5000 | L2 | concat | 0.7 | euclidean | 20 | 0.7262 | 0.7259 | 0.7548 | 0.7255±0.0040 | 0.7305±0.0042 |

Variant A is **at or below chance** — the frozen ShipSpotting config does not transfer; Variant B recovers
via re-tuning to **0.65–0.73**, best at the mixed `RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR)` detector
(**0.726**, though its RGB-OOD half still aids separation), then `IR · original→degraded` (**0.692**).

##### Variant-B ablations

_Vary one axis; others held at each detector's Variant-B config. Cells = val / test AUROC._

**IR · original→degraded** (Variant B: Lmid · concat · EV0.7 · euc · dk50)

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.7 · score=euclidean · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.659 | 0.656 | 0.627 | 0.647 | 0.668 | 0.645 | 0.656 | 0.713 |
| test AUROC | 0.648 | 0.645 | 0.621 | 0.643 | 0.666 | 0.649 | 0.649 | 0.692 |

_Vary **EV** · fixed: size=10000 · layer=Lmid · agg=concat · score=euclidean · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.500 | 0.713 | 0.328 | 0.291 | 0.280 | 0.280 |
| test AUROC | 0.500 | 0.692 | 0.317 | 0.281 | 0.271 | 0.272 |

_Vary **Score** · fixed: size=10000 · layer=Lmid · agg=concat · EV=0.7 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.222 | 0.704 | 0.130 | 0.713 |
| test AUROC | 0.233 | 0.680 | 0.140 | 0.692 |

_Vary **drop_k** · fixed: size=10000 · layer=Lmid · agg=concat · EV=0.7 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.501 | 0.498 | 0.600 | 0.713 | 0.500 |
| test AUROC | 0.512 | 0.492 | 0.591 | 0.692 | 0.500 |

**IR · degraded→degraded** (Variant B: L1 · mean · EV0.5 · euc · dk20)

_Vary **Layer** · fixed: size=1000 · agg=mean · EV=0.5 · score=euclidean · drop_k=20_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.647 | 0.642 | 0.637 | 0.632 | 0.633 | 0.582 | 0.398 | 0.500 |
| test AUROC | 0.649 | 0.649 | 0.655 | 0.653 | 0.647 | 0.577 | 0.395 | 0.492 |

_Vary **EV** · fixed: size=1000 · layer=L1 · agg=mean · score=euclidean · drop_k=20_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.647 | 0.611 | 0.549 | 0.531 | 0.513 | 0.510 |
| test AUROC | 0.649 | 0.610 | 0.548 | 0.531 | 0.513 | 0.510 |

_Vary **Score** · fixed: size=1000 · layer=L1 · agg=mean · EV=0.5 · drop_k=20_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.239 | 0.631 | 0.164 | 0.647 |
| test AUROC | 0.250 | 0.635 | 0.181 | 0.649 |

_Vary **drop_k** · fixed: size=1000 · layer=L1 · agg=mean · EV=0.5 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.484 | 0.603 | 0.647 | 0.567 | 0.500 |
| test AUROC | 0.482 | 0.611 | 0.649 | 0.577 | 0.500 |

**RGB+IR pooled · original(RGB),degraded(IR)→original(RGB),degraded(IR)** (Variant B: L2 · concat · EV0.7 · euc · dk20)

_Vary **Layer** · fixed: size=5000 · agg=concat · EV=0.7 · score=euclidean · drop_k=20_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.721 | 0.726 | 0.714 | 0.706 | 0.708 | 0.678 | 0.597 | 0.319 |
| test AUROC | 0.723 | 0.726 | 0.718 | 0.714 | 0.715 | 0.676 | 0.594 | 0.304 |

_Vary **EV** · fixed: size=5000 · layer=L2 · agg=concat · score=euclidean · drop_k=20_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.707 | 0.726 | 0.705 | 0.694 | 0.678 | 0.676 |
| test AUROC | 0.721 | 0.726 | 0.707 | 0.696 | 0.682 | 0.680 |

_Vary **Score** · fixed: size=5000 · layer=L2 · agg=concat · EV=0.7 · drop_k=20_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.335 | 0.715 | 0.261 | 0.726 |
| test AUROC | 0.326 | 0.712 | 0.256 | 0.726 |

_Vary **drop_k** · fixed: size=5000 · layer=L2 · agg=concat · EV=0.7 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.657 | 0.715 | 0.726 | 0.699 | 0.692 |
| test AUROC | 0.631 | 0.715 | 0.726 | 0.700 | 0.690 |

#### In-domain — fit & eval on Infiray (real)

##### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| real (raw) | all | L8 | mean | 0.999 | reconstruction | 0 | 0.7736 | 0.7323 | 0.5971 | 0.7302±0.0110 | 0.7624±0.0114 |
| enhanced (no-prompt) | all | Lmid | concat | 0.999 | reconstruction | 0 | 0.8235 | 0.7515 | 0.5081 | 0.7521±0.0119 | 0.7133±0.0143 |

##### Ablations

_Each table varies one axis; all others held at the experiment's val-best (listed per table). Cells = val / test AUROC._

**real (raw)**

_Vary **Layer** · fixed: size=all · agg=mean · EV=0.999 · score=reconstruction · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.766 | 0.770 | 0.771 | 0.773 | 0.774 | 0.771 | 0.768 | 0.716 |
| test AUROC | 0.722 | 0.725 | 0.724 | 0.728 | 0.732 | 0.732 | 0.728 | 0.656 |

_Vary **EV** · fixed: size=all · layer=L8 · agg=mean · score=reconstruction · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.710 | 0.729 | 0.752 | 0.760 | 0.754 | 0.774 |
| test AUROC | 0.669 | 0.682 | 0.703 | 0.716 | 0.712 | 0.732 |

_Vary **Score** · fixed: size=all · layer=L8 · agg=mean · EV=0.999 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.774 | 0.749 | 0.771 | 0.583 |
| test AUROC | 0.732 | 0.703 | 0.729 | 0.537 |

_Vary **drop_k** · fixed: size=all · layer=L8 · agg=mean · EV=0.999 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.774 | 0.223 | 0.208 | 0.336 | 0.442 |
| test AUROC | 0.732 | 0.203 | 0.222 | 0.349 | 0.421 |

**enhanced (no-prompt)**

_Vary **Layer** · fixed: size=all · agg=concat · EV=0.999 · score=reconstruction · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.690 | 0.690 | 0.716 | 0.705 | 0.743 | 0.753 | 0.793 | 0.823 |
| test AUROC | 0.646 | 0.653 | 0.697 | 0.677 | 0.702 | 0.699 | 0.729 | 0.751 |

_Vary **EV** · fixed: size=all · layer=Lmid · agg=concat · score=reconstruction · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.641 | 0.684 | 0.742 | 0.756 | 0.781 | 0.823 |
| test AUROC | 0.592 | 0.622 | 0.679 | 0.691 | 0.714 | 0.751 |

_Vary **Score** · fixed: size=all · layer=Lmid · agg=concat · EV=0.999 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.823 | 0.768 | 0.809 | 0.591 |
| test AUROC | 0.751 | 0.701 | 0.743 | 0.540 |

_Vary **drop_k** · fixed: size=all · layer=Lmid · agg=concat · EV=0.999 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.823 | 0.455 | 0.479 | 0.515 | 0.534 |
| test AUROC | 0.751 | 0.422 | 0.445 | 0.476 | 0.487 |

### Continual backbone

#### Cross-domain — ShipSpotting Continual detectors → Infiray (real)

**Variant A — frozen ShipSpotting val-best config**

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→degraded | 1000 | L8 | concat | 0.7 | reconstruction | 5 | 0.5079 | 0.4978 | 0.3562 | 0.4983±0.0076 | 0.4921±0.0054 |
| IR · degraded→degraded | 5000 | L6 | concat | 0.99 | reconstruction | 0 | 0.3827 | 0.3753 | 0.2850 | 0.3747±0.0100 | 0.4077±0.0052 |

**Variant B — config re-selected on Infiray val**

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→degraded | 1000 | L12 | mean | 0.7 | euclidean | 5 | 0.7026 | 0.6730 | 0.4914 | 0.6706±0.0089 | 0.6276±0.0110 |
| IR · degraded→degraded | 5000 | L8 | concat | 0.7 | euclidean | 0 | 0.6914 | 0.6811 | 0.4897 | 0.6771±0.0084 | 0.6214±0.0105 |

##### Variant-B ablations

_Vary one axis; others held at each detector's Variant-B config. Cells = val / test AUROC._

**IR · original→degraded** (Variant B: L12 · mean · EV0.7 · euc · dk5)

_Vary **Layer** · fixed: size=1000 · agg=mean · EV=0.7 · score=euclidean · drop_k=5_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.607 | 0.603 | 0.612 | 0.654 | 0.665 | 0.703 | 0.559 | 0.493 |
| test AUROC | 0.595 | 0.584 | 0.593 | 0.637 | 0.648 | 0.673 | 0.521 | 0.469 |

_Vary **EV** · fixed: size=1000 · layer=L12 · agg=mean · score=euclidean · drop_k=5_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.657 | 0.703 | 0.697 | 0.675 | 0.664 | 0.650 |
| test AUROC | 0.622 | 0.673 | 0.666 | 0.647 | 0.638 | 0.626 |

_Vary **Score** · fixed: size=1000 · layer=L12 · agg=mean · EV=0.7 · drop_k=5_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.452 | 0.692 | 0.307 | 0.703 |
| test AUROC | 0.441 | 0.676 | 0.317 | 0.673 |

_Vary **drop_k** · fixed: size=1000 · layer=L12 · agg=mean · EV=0.7 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.689 | 0.703 | 0.673 | 0.659 | 0.640 |
| test AUROC | 0.665 | 0.673 | 0.661 | 0.649 | 0.641 |

**IR · degraded→degraded** (Variant B: L8 · concat · EV0.7 · euc · dk0)

_Vary **Layer** · fixed: size=5000 · agg=concat · EV=0.7 · score=euclidean · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.603 | 0.604 | 0.639 | 0.679 | 0.691 | 0.650 | 0.214 | 0.016 |
| test AUROC | 0.590 | 0.586 | 0.625 | 0.667 | 0.681 | 0.636 | 0.208 | 0.016 |

_Vary **EV** · fixed: size=5000 · layer=L8 · agg=concat · score=euclidean · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.662 | 0.691 | 0.632 | 0.593 | 0.553 | 0.546 |
| test AUROC | 0.644 | 0.681 | 0.619 | 0.580 | 0.544 | 0.537 |

_Vary **Score** · fixed: size=5000 · layer=L8 · agg=concat · EV=0.7 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.298 | 0.671 | 0.190 | 0.691 |
| test AUROC | 0.267 | 0.672 | 0.172 | 0.681 |

_Vary **drop_k** · fixed: size=5000 · layer=L8 · agg=concat · EV=0.7 · score=euclidean_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.691 | 0.658 | 0.678 | 0.643 | 0.637 |
| test AUROC | 0.681 | 0.655 | 0.667 | 0.644 | 0.658 |

#### In-domain — fit & eval on Infiray (real)

##### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| real (raw) | all | L18 | concat | 0.999 | cosine | 0 | 0.9434 | 0.9254 | 0.8502 | 0.9240±0.0083 | 0.9247±0.0091 |
| enhanced (no-prompt) | all | L18 | concat | 0.999 | cosine | 0 | 0.8768 | 0.8589 | 0.7368 | 0.8577±0.0126 | 0.8601±0.0153 |

##### Ablations

_Each table varies one axis; all others held at the experiment's val-best. Cells = val / test AUROC._

**real (raw)**

_Vary **Layer** · fixed: size=all · agg=concat · EV=0.999 · score=cosine · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.921 | 0.720 | 0.848 | 0.887 | 0.913 | 0.929 | 0.943 | 0.913 |
| test AUROC | 0.900 | 0.709 | 0.835 | 0.879 | 0.902 | 0.915 | 0.925 | 0.870 |

_Vary **EV** · fixed: size=all · layer=L18 · agg=concat · score=cosine · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.851 | 0.886 | 0.922 | 0.930 | 0.940 | 0.943 |
| test AUROC | 0.812 | 0.845 | 0.892 | 0.899 | 0.914 | 0.925 |

_Vary **Score** · fixed: size=all · layer=L18 · agg=concat · EV=0.999 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.942 | 0.936 | 0.943 | 0.578 |
| test AUROC | 0.918 | 0.901 | 0.925 | 0.504 |

_Vary **drop_k** · fixed: size=all · layer=L18 · agg=concat · EV=0.999 · score=cosine_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.943 | 0.227 | 0.194 | 0.189 | 0.258 |
| test AUROC | 0.925 | 0.202 | 0.165 | 0.158 | 0.233 |

**enhanced (no-prompt)**

_Vary **Layer** · fixed: size=all · agg=concat · EV=0.999 · score=cosine · drop_k=0_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.849 | 0.670 | 0.763 | 0.817 | 0.846 | 0.862 | 0.877 | 0.848 |
| test AUROC | 0.821 | 0.698 | 0.773 | 0.813 | 0.839 | 0.848 | 0.859 | 0.807 |

_Vary **EV** · fixed: size=all · layer=L18 · agg=concat · score=cosine · drop_k=0_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.795 | 0.812 | 0.846 | 0.855 | 0.863 | 0.877 |
| test AUROC | 0.761 | 0.785 | 0.818 | 0.833 | 0.839 | 0.859 |

_Vary **Score** · fixed: size=all · layer=L18 · agg=concat · EV=0.999 · drop_k=0_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.873 | 0.855 | 0.877 | 0.556 |
| test AUROC | 0.853 | 0.828 | 0.859 | 0.536 |

_Vary **drop_k** · fixed: size=all · layer=L18 · agg=concat · EV=0.999 · score=cosine_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.877 | 0.217 | 0.158 | 0.177 | 0.227 |
| test AUROC | 0.859 | 0.222 | 0.183 | 0.208 | 0.240 |

### Analysis — Base vs Continual (Infiray, real)

| In-domain test AUROC | Base | Continual | Δ |
|:--|--:|--:|--:|
| real (raw) | 0.732 | **0.925** | **+0.193** |
| enhanced (no-prompt) | 0.751 | **0.859** | +0.108 |

**Findings (Infiray):**
- **Continual ≫ Base on a new IR dataset** (+0.19 on real): the IR-specialized backbone transfers; the
  RGB+IR base backbone does not generalize as well to unseen IR.
- **Enhancement helps Base, hurts Continual.** Enhanced-no-prompt lifts Base 0.732→0.751 but drops
  Continual 0.925→0.859 — the IR-native backbone prefers the unaltered real image.
- **Frozen ShipSpotting configs do not transfer** — every pure-IR Variant-A AUROC is below 0.5 (the
  mixed-OOD pooled detector sits near chance at 0.57). To a ShipSpotting-fit PCA, Infiray reconstructs
  *worse* than the degraded ShipSpotting OOD ships, so Infiray-ID scores higher than the positives and the
  ranking inverts (a **domain confound**). Re-tuning on Infiray val (Variant B) recovers only to **0.65–0.73**
  — and still compares two datasets. (Pure-original-OOD detectors are excluded: their sharp/RGB OOD vs
  blurry Infiray would let the score separate by resolution.)
- **drop_k must stay 0** — dropping any leading component collapses both backbones (the drop_k ablation
  falls 0.73→0.20 for Base, 0.93→0.20 for Continual). Both keep nearly all variance (EV 0.999).
- **Depth helps the Continual backbone** (L18 concat → 0.925); the Base backbone tops out shallower at a
  mean-pooled L8 (0.732). Euclidean scoring fails on Infiray for both (~0.5–0.54).

## Findings

1. **ESRGAN-augmented pretraining buys degradation robustness.** The +0.27 AUROC gap in
   degraded→degraded is the clearest signal: degradation is in the continual backbone's training
   distribution, so it represents degraded IR as in-distribution; the base backbone cannot.
2. **Base model's degraded→degraded is near-useless (0.59).** Fitting PCA on degraded features
   from a non-degradation-aware backbone does not recover — the subspace itself is corrupted.
3. **Recipe shifts with the backbone.** Base winners truncate hard (EV=0.5, latest-2 layers);
   continual winners keep EV=0.99, drop_k=0, mid layers (L6) — its IR-adapted features spread useful
   signal across more components. A full re-sweep per backbone was necessary.
4. **Reconstruction is the only scoring that works** in every experiment; the paper's mid-early
   `Lmid` layer default collapses to ~0.5–0.6 everywhere (it was tuned for defect localization).
5. **Size plateaus by ~1000–5000 images/class**; AUPR is base-rate-suppressed on the full 19%-OOD
   test (balanced AUPR is +0.25–0.34 higher).
6. **The Continual backbone also wins on a *different* IR dataset.** On Infiray (real), Continual reaches
   0.925 vs Base's 0.732 — degradation-aware SSL transfers across datasets, not just degradation levels.
7. **Detectors do not transfer frozen across datasets.** A ShipSpotting detector applied verbatim to
   Infiray scores *at or below chance* (domain confound: the new domain reconstructs worse than the source
   OOD); re-tuning the config on the target's val recovers it (to ~0.65–0.73). Backbone choice transfers;
   config choice does not.
8. **Domain-matched SSL ≫ off-the-shelf scale.** Off-the-shelf DINOv2 ViT-L (0.662) beats the Base
   backbone on degraded IR but trails Continual (0.860) by a wide margin, at 3.8× the inference cost.

## Reproduce

All commands in `../COMMAND.md`. One-shot (ShipSpotting): `scripts/maritime/run_all.sh`; balanced metrics:
`scripts/maritime/eval_balanced.py` (writes `tables/balanced_test.csv`). Infiray: `run_infiray.sh`
(in-domain) + `run_infcross.sh` (cross-domain); off-the-shelf: `run_hf.sh`. CSVs in `tables/`.
