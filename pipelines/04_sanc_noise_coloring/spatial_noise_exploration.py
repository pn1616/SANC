"""Pipeline 04 - Early SANC prototype (kept for the report's design history).

Before the FFT formulation, SANC was explored as *spatial amplitude* control:
    noise_map = BASE_NOISE * (1 - M)          (cells 175-182)
and as spatially smoothed shared-RGB noise (cell 183). These variants change
the noise *amplitude* per pixel and therefore break the unit-variance noise
assumption of DDPM; the final SANC instead changes the *spectrum* per pixel and
re-standardises (see sanc/noise/sanc.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import SANCMedicalDataset, build_combined_df, get_split  # noqa: E402
from sanc.models import load_saliency_model  # noqa: E402
from sanc.utils import ensure_dir, get_device, set_seed, tensor_to_image  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "04_sanc_noise_coloring"))
    args = ap.parse_args()
    set_seed(C.SEED)
    device, out = get_device(), ensure_dir(args.out)
    model = load_saliency_model(C.CKPT["saliency"], device)
    s = SANCMedicalDataset(get_split(build_combined_df(), "test", datasets=["Kvasir-SEG"]).head(1))[0]
    img = s["low_light"].unsqueeze(0).to(device)
    M = model(img)

    panels = [(tensor_to_image(img), "Low-light"), (M[0, 0].cpu(), "Saliency M")]
    for base in (0.05, 0.10, 0.15):
        noisy = (img + (base * (1 - M)).repeat(1, 3, 1, 1) * torch.randn_like(img)).clamp(0, 1)
        panels.append((tensor_to_image(noisy), f"amplitude BASE={base}"))
    for base in (0.02, 0.05):
        raw = torch.randn(1, 1, *img.shape[-2:], device=device)
        smooth = F.avg_pool2d(raw, 9, stride=1, padding=4)
        smooth = smooth / (smooth.std() + 1e-8)
        noisy = (img + (base * (1 - M) * smooth).repeat(1, 3, 1, 1)).clamp(0, 1)
        panels.append((tensor_to_image(noisy), f"smooth noise={base}"))

    fig, ax = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(out / "spatial_noise_prototype.png", dpi=110)
    print("Saved", out / "spatial_noise_prototype.png")


if __name__ == "__main__":
    main()
