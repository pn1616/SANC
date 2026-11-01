"""Pipeline 03 - Test-set evaluation of a saliency checkpoint, per dataset.

Uses per-image Dice / IoU at threshold 0.5 on the 409 valid test images
(the empty Kvasir mask is excluded). Reported result for Experiment A:

    HAM10000  130  Dice 0.8891  IoU 0.8180
    Kvasir    149  Dice 0.7565  IoU 0.6423
    ORIGA     130  Dice 0.9368  IoU 0.8848

Usage:
    python pipelines/03_saliency_unet/evaluate.py --arch unet --checkpoint $SANC_ROOT/models/best_unet_saliency.pth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from sanc import config as C  # noqa: E402
from sanc.data import build_combined_df, get_split, make_loader  # noqa: E402
from sanc.models import build_segmentation_model  # noqa: E402
from sanc.training import per_dataset_test_metrics  # noqa: E402
from sanc.utils import ensure_dir, get_device, load_checkpoint, state_dict_from  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet", choices=["unet", "attention_unet", "unetpp"])
    ap.add_argument("--checkpoint", default=str(C.CKPT["saliency"]))
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="experiment_A")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "03_saliency_unet"))
    args = ap.parse_args()

    device = get_device()
    out = ensure_dir(args.out)
    model = build_segmentation_model(args.arch).to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict_from(ckpt))
    print(f"Loaded {args.checkpoint} | epoch {ckpt.get('epoch')} | val Dice {ckpt.get('val_dice')}")

    df = get_split(build_combined_df(), args.split)
    per_image, summary = per_dataset_test_metrics(model, make_loader(df), device, logits=True, threshold=args.threshold)
    overall = per_image[["dice", "iou"]].mean()
    print(summary.round(4).to_string(index=False))
    print(f"\nOverall ({len(per_image)} images): Dice {overall['dice']:.4f} | IoU {overall['iou']:.4f}")
    per_image.to_csv(out / f"saliency_{args.split}_per_image_{args.tag}.csv", index=False)
    summary.to_csv(out / f"saliency_{args.split}_results_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
