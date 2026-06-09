# Maritime OOD — Results

All quantitative results, grouped by backbone. Each model part (1 & 2) carries the **same set of
tables**: Summary · Balanced-test detail · Size sweep · Ablations. Part 3 compares the two backbones.
Method is in `ANOMALY_CLASS.md`; run commands in `../COMMAND.md`.

## Overview

Eval setup (both backbones): CLS token, 224 px, normalization `(0.5,0.5,0.5)`. Selection on **val AUROC**;
test reported on the val-selected config. Degraded data = Real-ESRGAN USM degradation of the IR set.

Both are the **same ViT-L/16 architecture** — only the SSL checkpoint differs:

| Variant | Checkpoint | SSL pretraining |
|---|---|---|
| **Base** | `ViT-L-16/eval/training_2348399` | self-supervised on **original RGB + IR** |
| **Continual** | `ViT-L-16-Continual-IR/eval/training_51199` | **initialized from Base**, then continued SSL on **IR with ESRGAN degradation mixed in** (`esrgan_prob=0.8`) |

Eval splits (per condition): full test = 1700 ID + 400 OOD; balanced test = 400 ID + 400 OOD (20-seed,
val-derived threshold held fixed). "original→degraded" = PCA fit on original images, evaluated on degraded;
"degraded→degraded" = fit and evaluated on degraded. Pooled = RGB and IR treated as separate samples
under one PCA. Artifacts (this folder): `grid_summary_all.csv` (66,709 configs), `best_per_experiment.csv`,
`balanced_test.csv`.

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

| Experiment | agg · layers | D | EV | pca_k | drop_k | **k_eff = pca_k − dk** |
|:--|:--|--:|--:|--:|--:|--:|
| **Base** — RGB · original→original | concat · L2 | 2048 | 0.50 | 128 | 20 | **108** |
| **Base** — IR · original→original | concat · L2 | 2048 | 0.50 | 126 | 50 | **76** |
| **Base** — IR · original→degraded | concat · L8 | 8192 | 0.70 | 211 | 50 | **161** |
| **Base** — IR · degraded→degraded | mean · L4 | 1024 | 0.99 | 768 | 5 | **763** |
| **Base** — RGB+IR pooled · original→original | concat · L2 | 2048 | 0.50 | 131 | 50 | **81** |
| **Continual** — IR · original→original | concat · L6 | 6144 | 0.99 | 954 | 0 | **954** |
| **Continual** — IR · original→degraded | concat · L8 | 8192 | 0.70 | 189 | 5 | **184** |
| **Continual** — IR · degraded→degraded | concat · L6 | 6144 | 0.99 | 970 | 0 | **970** |

**Worked example** — RGB · original→original (`concat · L2`, EV=0.50, drop_k=20):
`D = 2 × 1024 = 2048`. PCA eigenvalues have no dominant direction (top = 1.57% of variance); walking
the cumulative ratio `cumvar(127)=0.4998 (<0.50) → cumvar(128)=0.5021 (≥0.50)` gives **pca_k = 128**.
Dropping the top 20 leaves **k_eff = 108** components reconstructing “normal”; the score is the residual
energy in the other `2048 − 108 = 1940` directions (top-20 nuisance head + 1920-dim tail).

The split across backbones is stark: the **base** winners truncate hard (k_eff ≈ 80–110 of 2048),
isolating a narrow mid-spectrum band; the **continual** winners use k_eff ≈ 950 of 6144 (drop_k=0) —
their IR-adapted features spread discriminative variance across a much higher-rank subspace.


# Part 1 — Base backbone

### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| RGB · original→original | 10000 | L2 | concat | 0.5 | reconstruction | 20 | 0.8664 | 0.8773 | 0.5440 | 0.8793±0.0087 | 0.8347±0.0159 |
| IR · original→original | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8340 | 0.8217 | 0.4259 | 0.8276±0.0101 | 0.7671±0.0156 |
| IR · original→degraded | 10000 | L8 | concat | 0.7 | reconstruction | 50 | 0.7155 | 0.7104 | 0.3175 | 0.7129±0.0141 | 0.6664±0.0165 |
| IR · degraded→degraded | 1000 | L4 | mean | 0.99 | reconstruction | 5 | 0.5932 | 0.5878 | 0.2388 | 0.5882±0.0104 | 0.5676±0.0107 |
| RGB+IR pooled · original→original | 10000 | L2 | concat | 0.5 | reconstruction | 50 | 0.8544 | 0.8547 | 0.5021 | 0.8558±0.0066 | 0.8042±0.0107 |

### Balanced-test detail (400 ID + 400 OOD, 20-seed mean)

| Experiment | accuracy ↑ | OOD precision ↑ | OOD recall ↑ | OOD F1 ↑ |
|:--|--:|--:|--:|--:|
| RGB · original→original | 0.7904 | 0.7983 | 0.7775 | 0.7877 |
| IR · original→original | 0.7583 | 0.7204 | 0.8450 | 0.7777 |
| IR · original→degraded | 0.6558 | 0.6153 | 0.8325 | 0.7076 |
| IR · degraded→degraded | 0.5748 | 0.5844 | 0.5200 | 0.5502 |
| RGB+IR pooled · original→original | 0.7787 | 0.7581 | 0.8187 | 0.7872 |

### Size sweep

| fit size | RGB · original→original | IR · original→original | IR · original→degraded | IR · degraded→degraded | RGB+IR pooled · original→original |
|--:|--:|--:|--:|--:|--:|
| 100 | 0.838 / 0.857 | 0.817 / 0.822 | 0.699 / 0.702 | 0.585 / 0.579 | 0.821 / 0.834 |
| 500 | 0.861 / 0.865 | 0.821 / 0.823 | 0.709 / 0.712 | 0.592 / 0.587 | 0.846 / 0.849 |
| 1000 | 0.853 / 0.860 | 0.828 / 0.820 | 0.709 / 0.708 | 0.593 / 0.588 | 0.846 / 0.846 |
| 5000 | 0.864 / 0.867 | 0.833 / 0.822 | 0.709 / 0.695 | 0.590 / 0.580 | 0.852 / 0.854 |
| 10000 | 0.866 / 0.877 | 0.834 / 0.822 | 0.715 / 0.710 | 0.591 / 0.576 | 0.854 / 0.855 |

_(val / test AUROC; val-best config at each fit size)_

### Ablations

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

_Vary **Layer** · fixed: size=10000 · agg=concat · EV=0.5 · score=reconstruction · drop_k=50_
| Layer | L1 | L2 | L4 | L6 | L8 | L12 | L18 | Lmid |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.766 | 0.854 | 0.784 | 0.754 | 0.779 | 0.767 | 0.707 | 0.557 |
| test AUROC | 0.766 | 0.855 | 0.796 | 0.772 | 0.788 | 0.779 | 0.713 | 0.551 |

_Vary **EV** · fixed: size=10000 · layers=L2 · agg=concat · score=reconstruction · drop_k=50_
| EV | 0.5 | 0.7 | 0.9 | 0.95 | 0.99 | 0.999 |
|:--|--:|--:|--:|--:|--:|--:|
| val AUROC | 0.854 | 0.787 | 0.580 | 0.522 | 0.480 | 0.472 |
| test AUROC | 0.855 | 0.783 | 0.572 | 0.512 | 0.468 | 0.459 |

_Vary **Score** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · drop_k=50_
| Score | reconstruction | mahalanobis | cosine | euclidean |
|:--|--:|--:|--:|--:|
| val AUROC | 0.854 | 0.547 | 0.712 | 0.539 |
| test AUROC | 0.855 | 0.550 | 0.705 | 0.541 |

_Vary **drop_k** · fixed: size=10000 · layers=L2 · agg=concat · EV=0.5 · score=reconstruction_
| drop_k | 0 | 5 | 20 | 50 | 100 |
|:--|--:|--:|--:|--:|--:|
| val AUROC | 0.788 | 0.809 | 0.836 | 0.854 | 0.824 |
| test AUROC | 0.800 | 0.819 | 0.844 | 0.855 | 0.825 |


# Part 2 — Continual backbone

### Summary

| Experiment | size | layers | agg | EV | score | dk | val AUROC ↑ | test AUROC ↑ | test AUPR ↑ | bal AUROC ↑ | bal AUPR ↑ |
|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| IR · original→original | 5000 | L6 | concat | 0.99 | reconstruction | 0 | 0.8703 | 0.8814 | 0.5784 | 0.8820±0.0077 | 0.8458±0.0153 |
| IR · original→degraded | 1000 | L8 | concat | 0.7 | reconstruction | 5 | 0.8371 | 0.8358 | 0.5104 | 0.8358±0.0100 | 0.8067±0.0159 |
| IR · degraded→degraded | 5000 | L6 | concat | 0.99 | reconstruction | 0 | 0.8516 | 0.8600 | 0.5074 | 0.8608±0.0092 | 0.8091±0.0176 |

### Balanced-test detail (400 ID + 400 OOD, 20-seed mean)

| Experiment | accuracy ↑ | OOD precision ↑ | OOD recall ↑ | OOD F1 ↑ |
|:--|--:|--:|--:|--:|
| IR · original→original | 0.8086 | 0.8364 | 0.7675 | 0.8005 |
| IR · original→degraded | 0.7425 | 0.7683 | 0.6950 | 0.7297 |
| IR · degraded→degraded | 0.7925 | 0.7637 | 0.8475 | 0.8034 |

### Size sweep

| fit size | IR · original→original | IR · original→degraded | IR · degraded→degraded |
|--:|--:|--:|--:|
| 100 | 0.821 / 0.821 | 0.802 / 0.800 | 0.775 / 0.769 |
| 500 | 0.869 / 0.879 | 0.833 / 0.832 | 0.850 / 0.853 |
| 1000 | 0.869 / 0.879 | 0.837 / 0.836 | 0.851 / 0.857 |
| 5000 | 0.870 / 0.881 | 0.834 / 0.834 | 0.852 / 0.860 |
| 10000 | 0.868 / 0.882 | 0.834 / 0.834 | 0.851 / 0.860 |

_(val / test AUROC; val-best config at each fit size)_

### Ablations

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


# Part 3 — Base vs Continual (IR robustness)

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

## Reproduce

All commands in `../COMMAND.md`. One-shot: `scripts/maritime/run_all.sh`;
balanced metrics: `scripts/maritime/eval_balanced.py` (writes `balanced_test.csv`).
