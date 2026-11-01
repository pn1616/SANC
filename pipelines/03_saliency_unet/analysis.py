"""Pipeline 03 - Post-hoc analyses of the selected saliency U-Net.

Sub-commands
  thresholds  continuous saliency: soft Dice + Dice/IoU at thresholds 0.2..0.8 (cell 168)
  errors      Kvasir-SEG over/under-segmentation analysis (cell 169)
  refine      Experiment G: connected-component cleanup + blur, no retraining (cells 171-172)
              -> Dice 0.7565 -> 0.7543 on Kvasir test (no improvement, not adopted)

Usage:
    python pipelines/03_saliency_unet/analysis.py thresholds
    python pipelines/03_saliency_unet/analysis.py errors
    python pipelines/03_saliency_unet/analysis.py refine
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import SANCMedicalDataset, build_combined_df, get_split, make_loader  # noqa: E402
from sanc.metrics import soft_dice  # noqa: E402
from sanc.models import load_saliency_model  # noqa: E402
from sanc.utils import ensure_dir, get_device  # noqa: E402


def refine_saliency(prob, threshold=0.5, min_component_size=20, blur_kernel=3):
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned[labels == lab] = 1
    refined = (prob * cleaned).astype(np.float32)
    if blur_kernel > 1:
        refined = cv2.GaussianBlur(refined, (blur_kernel, blur_kernel), 0)
    return refined


def confusion(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    return tp, fp, fn, 2 * tp / (2 * tp + fp + fn + 1e-8), tp / (tp + fp + fn + 1e-8)


@torch.no_grad()
def cmd_thresholds(model, device, out):
    loader = make_loader(get_split(build_combined_df(), "val"))
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    res = {t: [] for t in ths}; soft = []
    for b in loader:
        x, y = b["low_light"].to(device), b["mask"].to(device)
        s = model(x)
        soft += soft_dice(s, y).cpu().tolist()
        for t in ths:
            p = (s >= t).float()
            inter = (p * y).sum((1, 2, 3)); ps, ys = p.sum((1, 2, 3)), y.sum((1, 2, 3))
            d = (2 * inter + 1e-6) / (ps + ys + 1e-6); j = (inter + 1e-6) / (ps + ys - inter + 1e-6)
            res[t] += list(zip(d.cpu().tolist(), j.cpu().tolist()))
    table = pd.DataFrame([dict(threshold=t, dice=np.mean([a for a, _ in v]), iou=np.mean([b for _, b in v]))
                          for t, v in res.items()])
    print(f"Soft Dice: {np.mean(soft):.4f}\n", table.round(4).to_string(index=False))
    table.to_csv(out / "threshold_sweep_val.csv", index=False)


@torch.no_grad()
def cmd_errors(model, device, out):
    df = get_split(build_combined_df(), "test", datasets=["Kvasir-SEG"])
    rows = []
    for b in make_loader(df):
        prob = model(b["low_light"].to(device)).cpu()
        for i in range(prob.shape[0]):
            gt, pred = b["mask"][i, 0].numpy(), (prob[i, 0].numpy() >= 0.5).astype(np.uint8)
            tp, fp, fn, dice, iou = confusion(pred, gt)
            g = int(gt.sum())
            kind = ("GOOD" if dice >= 0.70 else "OVERSEGMENTATION" if fp > fn * 1.25
                    else "UNDERSEGMENTATION" if fn > fp * 1.25 else "MIXED")
            rows.append(dict(filename=b["filename"][i], dice=dice, iou=iou, gt_pixels=g, pred_pixels=int(pred.sum()),
                             fp=fp, fn=fn, fp_ratio=fp / (g + 1e-8), fn_ratio=fn / (g + 1e-8), error_type=kind))
    r = pd.DataFrame(rows)
    print(r["error_type"].value_counts().to_string())
    print(r[["dice", "iou", "fp", "fn", "fp_ratio", "fn_ratio"]].mean().round(4).to_string())
    print("\nWorst over-segmentation:\n", r.nlargest(10, "fp")[["filename", "dice", "gt_pixels", "pred_pixels", "fp", "fn"]].to_string(index=False))
    print("\nWorst under-segmentation:\n", r.nlargest(10, "fn")[["filename", "dice", "gt_pixels", "pred_pixels", "fp", "fn"]].to_string(index=False))
    r.to_csv(out / "kvasir_error_analysis.csv", index=False)


@torch.no_grad()
def cmd_refine(model, device, out):
    ds = SANCMedicalDataset(get_split(build_combined_df(), "test", datasets=["Kvasir-SEG"]))
    od, oi, rd, ri = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        prob = model(s["low_light"].unsqueeze(0).to(device))[0, 0].cpu().numpy()
        gt = (s["mask"][0].numpy() >= 0.5).astype(np.uint8)
        *_, d, j = confusion((prob >= 0.5).astype(np.uint8), gt); od.append(d); oi.append(j)
        *_, d, j = confusion((refine_saliency(prob) >= 0.5).astype(np.uint8), gt); rd.append(d); ri.append(j)
    t = pd.DataFrame([dict(variant="original", dice=np.mean(od), iou=np.mean(oi)),
                      dict(variant="refined", dice=np.mean(rd), iou=np.mean(ri))])
    print(f"Valid samples: {len(ds)}\n", t.round(4).to_string(index=False))
    print(f"Dice change: {np.mean(rd) - np.mean(od):+.4f} | IoU change: {np.mean(ri) - np.mean(oi):+.4f}")
    t.to_csv(out / "experiment_G_refinement.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["thresholds", "errors", "refine"])
    ap.add_argument("--checkpoint", default=str(C.CKPT["saliency"]))
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "03_saliency_unet"))
    args = ap.parse_args()
    device = get_device()
    model = load_saliency_model(args.checkpoint, device, return_probs=True)
    {"thresholds": cmd_thresholds, "errors": cmd_errors, "refine": cmd_refine}[args.command](model, device, ensure_dir(args.out))


if __name__ == "__main__":
    main()
