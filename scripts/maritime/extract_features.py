"""Unified DINOv2 CLS feature extraction — one extractor, any backbone, any dataset.

Backbone (pick one):
  --backbone local : local DINOv2 checkpoint (--ckpt --config --submodule)
  --backbone hf    : off-the-shelf HF DINOv2 (--hf_model, e.g. facebook/dinov2-small)

Dataset = one image folder per split (ID = normal, OOD = anomaly):
  --id_fit  --id_val --ood_val  --id_test --ood_test

Writes a cache (fit/val/test CLS + labels) consumed by anomaly_detection.py.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import extract_per_layer_cls, list_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_local(args):
    """Local DINOv2 checkpoint → (forward_batch, depth, positives_sorted)."""
    from subspacead.core.extractor_local import LocalDinoV2Extractor
    extractor = LocalDinoV2Extractor(
        config_file=args.config, pretrained_weights=args.ckpt,
        opts=args.opts, submodule_path=args.submodule,
        norm_mean=tuple(args.norm_mean), norm_std=tuple(args.norm_std),
    )
    depth = (len(extractor.model.blocks[-1]) if extractor.model.chunked_blocks
             else len(extractor.model.blocks))
    positives = list(range(depth))   # every transformer layer (CLS token only)

    @torch.no_grad()
    def forward_batch(pils):
        x = extractor._preprocess(pils, args.image_res)
        # return_class_token=True → we keep the per-layer CLS token (image-level signal)
        outs = extractor.model.get_intermediate_layers(
            x, n=positives, reshape=False, return_class_token=True, norm=True)
        return torch.stack([c for (_p, c) in outs], dim=1)   # [B, depth, D] CLS per layer
    return forward_batch, depth, positives


def build_hf(args):
    """Off-the-shelf HF DINOv2 → (forward_batch, depth, positives_sorted)."""
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.hf_model)
    model = AutoModel.from_pretrained(args.hf_model).eval().to(DEVICE)
    depth = model.config.num_hidden_layers
    positives = list(range(depth))

    @torch.no_grad()
    def forward_batch(pils):
        inp = proc(images=pils, return_tensors="pt", do_resize=True,
                   size={"height": args.image_res, "width": args.image_res},
                   do_center_crop=False).to(DEVICE)
        out = model(**inp, output_hidden_states=True, output_attentions=False)
        # h[:, 0, :] = CLS token of each layer (image-level signal) → [B, depth, D]
        return torch.stack([h[:, 0, :] for h in out.hidden_states[1:]], dim=1)
    return forward_batch, depth, positives


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["local", "hf"])
    # local backbone
    p.add_argument("--ckpt", help="local DINOv2 SSL checkpoint (.pth)")
    p.add_argument("--config", help="DINOv2 config yaml")
    p.add_argument("--submodule", help="DINOv2 source tree")
    p.add_argument("--norm_mean", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    p.add_argument("--norm_std", type=float, nargs=3, default=[0.5, 0.5, 0.5])

    # hf backbone
    p.add_argument("--hf_model", help="e.g. facebook/dinov2-small")

    # dataset: explicit folder per split
    p.add_argument("--id_fit", required=True, help="ID images for PCA fit")
    p.add_argument("--id_val", required=True, help="ID validation images")
    p.add_argument("--id_test", required=True, help="ID test images")
    p.add_argument("--ood_val", required=True, help="OOD validation images")
    p.add_argument("--ood_test", required=True, help="OOD test images")
    p.add_argument("--out_cache_file", type=Path, required=True)
    p.add_argument("--image_res", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)

    # extra DINOv2 opts (local backbone only)
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[],
                   help="extra DINOv2 student.* opts for --backbone local")

    args = p.parse_args()

    if args.backbone == "local":
        if not (args.ckpt and args.config and args.submodule):
            p.error("--backbone local requires --ckpt --config --submodule")
        forward_batch, depth, positives = build_local(args)
    else:
        if not args.hf_model:
            p.error("--backbone hf requires --hf_model")
        forward_batch, depth, positives = build_hf(args)
    print(f"backbone={args.backbone} depth={depth} D-layers={len(positives)}")

    def extract(paths, desc):
        return extract_per_layer_cls(paths, args.batch_size, desc, forward_batch)

    # fit = ID only (PCA is fit on normal data)
    fit_cls, _ = extract(list_images(args.id_fit), "fit")

    # val / test = ID (label 0) + OOD (label 1)
    def split(id_dir, ood_dir, desc):
        idp, oodp = list_images(id_dir), list_images(ood_dir)
        labels = np.array([0] * len(idp) + [1] * len(oodp), dtype=np.int64)
        cls, valid = extract(idp + oodp, desc)
        return cls, valid, labels

    val_cls, val_valid, val_labels = split(args.id_val, args.ood_val, "val")
    test_cls, test_valid, test_labels = split(args.id_test, args.ood_test, "test")

    args.out_cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache_file,
        fit_cls=fit_cls,
        val_cls=val_cls, val_valid=val_valid, val_labels=val_labels,
        test_cls=test_cls, test_valid=test_valid, test_labels=test_labels,
        depth=np.array(depth, dtype=np.int32),
        positives_sorted=np.array(positives, dtype=np.int32),
        image_res=np.array(args.image_res, dtype=np.int32),
    )
    print(
        f"saved {args.out_cache_file} — fit {fit_cls.shape}, "
        f"val {val_cls.shape} (OOD {int(val_labels.sum())}/{len(val_labels)}), "
        f"test {test_cls.shape} (OOD {int(test_labels.sum())}/{len(test_labels)})")


if __name__ == "__main__":
    main()
