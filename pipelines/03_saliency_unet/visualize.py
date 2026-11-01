"""Pipeline 03 - Saliency figures: prediction grid (cell 140) and gradient saliency (cell 141)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import SANCMedicalDataset, build_combined_df, get_split  # noqa: E402
from sanc.models import load_saliency_model  # noqa: E402
from sanc.utils import ensure_dir, get_device  # noqa: E402


def gradient_saliency(model_logits, image, device):
    """|d(mean foreground logit)/d input|, max over RGB, min-max normalised."""
    x = image.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    logits = model_logits(x)
    pred = (torch.sigmoid(logits) > 0.5).float()
    if pred.sum() == 0:
        flat = logits.view(-1)
        target = torch.topk(flat, max(1, int(flat.numel() * 0.01))).values.mean()
    else:
        target = (logits * pred).sum() / (pred.sum() + 1e-8)
    model_logits.zero_grad(set_to_none=True)
    target.backward()
    s = x.grad.detach().abs().max(dim=1)[0][0]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy(), pred[0, 0].cpu().numpy(), s.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.CKPT["saliency"]))
    ap.add_argument("--per-dataset", type=int, default=3)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "03_saliency_unet"))
    args = ap.parse_args()
    device, out = get_device(), ensure_dir(args.out)
    model = load_saliency_model(args.checkpoint, device, return_probs=False)
    df = get_split(build_combined_df(), "test")

    samples = []
    for name in C.DATASET_NAMES:
        ds = SANCMedicalDataset(df[df["dataset"] == name])
        samples += [ds[i] for i in range(min(args.per_dataset, len(ds)))]

    titles = ["Original", "Low-light", "Ground truth", "Prediction", "Probability", "Gradient saliency"]
    fig, axes = plt.subplots(len(samples), 6, figsize=(18, 3 * len(samples)), squeeze=False)
    for r, s in enumerate(samples):
        prob, pred, grad = gradient_saliency(model, s["low_light"], device)
        imgs = [s["original"].permute(1, 2, 0).numpy(), s["low_light"].permute(1, 2, 0).numpy(),
                s["mask"][0].numpy(), pred, prob, grad]
        for c, (im, t) in enumerate(zip(imgs, titles)):
            axes[r, c].imshow(np.clip(im, 0, 1), cmap=None if c < 2 else ("hot" if c >= 4 else "gray"))
            axes[r, c].set_title(f"{s['dataset']}\n{t}" if c == 0 else t)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out / "saliency_predictions_and_gradients.png", dpi=110)
    print("Saved", out / "saliency_predictions_and_gradients.png")


if __name__ == "__main__":
    main()
