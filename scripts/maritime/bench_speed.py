"""Speed comparison for DINOv2 ViT-S / ViT-B / ViT-L, split by inference stage.

Per backbone, the per-image inference cost is broken into two stages and their total:
  1. Feature extraction (GPU): preprocess + DINOv2 forward (output_hidden_states) +
     stack per-layer CLS + transfer to host  -> [N, depth, D].
  2. SubspaceAD scoring (CPU/numpy): aggregate_features (L2 concat) + reconstruction
     residual via calculate_anomaly_scores, with a PCA fitted offline (not timed).
  3. Total = extraction + scoring (sequential).

Reported at batch=16 (throughput, img/s) and batch=1 (latency, ms/img), plus the
offline PCA-fit time (informational — training cost, not inference). The scoring config
is the fixed reference L2 concat - EV=0.5 - reconstruction - drop_k=20 (same as before),
so the SubspaceAD cost differs across models only via embed dim D = 2*embed.
Same GPU, 224 px, FP32. Writes results_maritime/tables/speed_bench.csv.
"""
import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores
from _common import _chunked, aggregate_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODELS = {"ViT-S": "facebook/dinov2-small", "ViT-B": "facebook/dinov2-base", "ViT-L": "facebook/dinov2-large"}
IMG_GLOB = "/data/hanchong/maritime-vessel-dataset-infrared-flux2-klein-degraded/In-distribution/test/*.jpg"

# Fixed reference SubspaceAD config (same as the previous pca-timing): L2 concat, EV0.5, reconstruction, dk20.
REF_LAYERS = [-1, -2]      # "L2" = last two transformer layers
REF_EV = 0.5
REF_SCORE = "reconstruction"
REF_DROP_K = 20


def _prep(proc, pils, res):
    return proc(images=pils, return_tensors="pt", do_resize=True,
                size={"height": res, "width": res}, do_center_crop=False).to(DEVICE)


@torch.no_grad()
def extract_with_timing(model, proc, pils, res, batch):
    """Time feature extraction (forward + CLS stack + host transfer) and return the CLS array.

    Returns (cls_all [N, depth, D] float32, img_per_s, ms_per_img).
    """
    # warmup
    w = _prep(proc, pils[:batch], res)
    for _ in range(2):
        model(**w, output_hidden_states=True, output_attentions=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    chunks, n = [], 0
    t0 = time.perf_counter()
    for i in range(0, len(pils), batch):
        inp = _prep(proc, pils[i:i + batch], res)
        out = model(**inp, output_hidden_states=True, output_attentions=False)
        cls = torch.stack([h[:, 0, :] for h in out.hidden_states[1:]], dim=1)  # [B, depth, D]
        chunks.append(cls.cpu())
        n += cls.shape[0]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    cls_all = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return cls_all, n / dt, 1000 * dt / n


@torch.no_grad()
def extract_latency(model, proc, pils, res, reps=100):
    """Single-image (batch=1) feature-extraction latency, ms/img."""
    one = _prep(proc, pils[:1], res)
    for _ in range(10):
        model(**one, output_hidden_states=True, output_attentions=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(reps):
        inp = _prep(proc, pils[i % len(pils):i % len(pils) + 1], res)
        out = model(**inp, output_hidden_states=True, output_attentions=False)
        _ = torch.stack([h[:, 0, :] for h in out.hidden_states[1:]], dim=1).cpu()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def fit_reference_pca(embed, n_fit=10000):
    """Offline PCA fit on synthetic [n_fit, 2*embed] (L2 concat). Returns (pca, fit_seconds)."""
    D = 2 * embed
    rng = np.random.default_rng(0)
    fit = rng.standard_normal((n_fit, D)).astype(np.float32)
    chunk = 1024
    t0 = time.perf_counter()
    pca = PCAModel(ev=REF_EV).fit(_chunked(fit, chunk), D, n_fit, (n_fit + chunk - 1) // chunk)
    return pca, time.perf_counter() - t0


def score_throughput(cls_all, depth, pca, batch):
    """SubspaceAD scoring throughput over the extracted features (CPU). Returns (img_per_s, ms_per_img)."""
    pos = list(range(depth))
    X = aggregate_features(cls_all, pos, REF_LAYERS, depth, "concat")  # [N, 2*embed]
    # warmup
    calculate_anomaly_scores(X[:batch], pca, REF_SCORE, drop_k=REF_DROP_K)
    n = 0
    t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        calculate_anomaly_scores(X[i:i + batch], pca, REF_SCORE, drop_k=REF_DROP_K)
        n += min(batch, len(X) - i)
    dt = time.perf_counter() - t0
    return n / dt, 1000 * dt / n


def score_latency(cls_all, depth, pca, reps=200):
    """Single-feature (batch=1) SubspaceAD scoring latency, ms/img (CPU)."""
    pos = list(range(depth))
    X = aggregate_features(cls_all[:1], pos, REF_LAYERS, depth, "concat")  # [1, 2*embed]
    for _ in range(5):
        calculate_anomaly_scores(X, pca, REF_SCORE, drop_k=REF_DROP_K)
    t0 = time.perf_counter()
    for _ in range(reps):
        calculate_anomaly_scores(X, pca, REF_SCORE, drop_k=REF_DROP_K)
    return 1000 * (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_imgs", type=int, default=320)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    paths = sorted(glob.glob(IMG_GLOB))[:args.n_imgs]
    pils = [Image.open(p).convert("RGB") for p in paths]
    print(f"benchmark on {len(pils)} imgs, res={args.res}, batch={args.batch}, device={DEVICE}\n")

    rows = []
    for name, ckpt in HF_MODELS.items():
        proc = AutoImageProcessor.from_pretrained(ckpt)
        model = AutoModel.from_pretrained(ckpt).eval().to(DEVICE)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        embed = model.config.hidden_size
        layers = model.config.num_hidden_layers
        patch = model.config.patch_size

        # Stage 1 — feature extraction (GPU)
        cls_all, ext_ips, ext_ms = extract_with_timing(model, proc, pils, args.res, args.batch)
        ext_lat = extract_latency(model, proc, pils, args.res)
        depth = cls_all.shape[1]

        # Stage 2 — SubspaceAD scoring (CPU); PCA fit offline (not timed into scoring)
        pca, pca_fit_s = fit_reference_pca(embed)
        ss_ips, ss_ms = score_throughput(cls_all, depth, pca, args.batch)
        ss_lat = score_latency(cls_all, depth, pca)

        # Stage 3 — total (sequential)
        tot_ms = ext_ms + ss_ms
        tot_ips = 1000.0 / tot_ms
        tot_lat = ext_lat + ss_lat

        rows.append({
            "backbone": name, "ckpt": ckpt, "params_M": round(params, 1),
            "embed": embed, "layers": layers, "patch": patch,
            "extract_img_per_s_b16": round(ext_ips, 1),
            "subspacead_img_per_s_b16": round(ss_ips, 1),
            "total_img_per_s_b16": round(tot_ips, 1),
            "extract_lat_ms_b1": round(ext_lat, 3),
            "subspacead_lat_ms_b1": round(ss_lat, 3),
            "total_lat_ms_b1": round(tot_lat, 3),
            "pca_fit_s": round(pca_fit_s, 2),
        })
        print(f"{name:6} params={params:6.1f}M embed={embed} layers={layers}\n"
              f"   throughput img/s (b16): extract={ext_ips:7.1f}  subspacead={ss_ips:9.1f}  total={tot_ips:7.1f}\n"
              f"   latency ms/img (b1):    extract={ext_lat:7.2f}  subspacead={ss_lat:9.4f}  total={tot_lat:7.2f}\n"
              f"   pca fit (offline)={pca_fit_s:.2f}s")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out = Path("/home/hcchua/SubspaceAD/results_maritime/tables/speed_bench.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
