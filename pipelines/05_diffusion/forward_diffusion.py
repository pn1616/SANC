"""Pipeline 05 - SANC-coloured forward diffusion on a real image (cells 188-189, 205-206).

x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps_SANC, linear schedule T=100, beta 1e-4..0.02.
Prints x_t statistics and saves the trajectory figure for t in {0,10,25,50,75,99}.
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

from sanc import config as C  # noqa: E402
from sanc.data import SANCMedicalDataset, build_combined_df, get_split  # noqa: E402
from sanc.diffusion import DiffusionSchedule  # noqa: E402
from sanc.models import load_saliency_model  # noqa: E402
from sanc.noise import SANCNoiseColoring  # noqa: E402
from sanc.utils import ensure_dir, get_device, set_seed, tensor_to_image  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Kvasir-SEG", choices=C.DATASET_NAMES)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "05_diffusion"))
    args = ap.parse_args()
    set_seed(C.SEED)
    device, out = get_device(), ensure_dir(args.out)
    sched = DiffusionSchedule(device=device)
    sanc = SANCNoiseColoring(norm_eps=1e-6).to(device)
    model = load_saliency_model(C.CKPT["saliency"], device)

    s = SANCMedicalDataset(get_split(build_combined_df(), "test", datasets=[args.dataset]).head(1))[0]
    x0, low = s["original"].unsqueeze(0).to(device), s["low_light"].unsqueeze(0).to(device)
    M = model(low)                                  # saliency predicted from the LOW-LIGHT image
    eps = sanc(torch.randn_like(x0), M)

    steps = [0, 10, 25, 50, 75, 99]
    traj = {}
    for t in steps:
        xt = sched.q_sample(x0, torch.tensor([t], device=device), eps)
        traj[t] = xt
        print(f"t={t:2d} | min={xt.min():.4f} | max={xt.max():.4f} | mean={xt.mean():.4f} | std={xt.std():.4f}")

    fig, ax = plt.subplots(2, 6, figsize=(22, 8))
    top = [(tensor_to_image(x0), "Original $x_0$"), (tensor_to_image(low), "Low-light"),
           (M[0, 0].cpu(), "Predicted saliency"), (tensor_to_image((eps - eps.min()) / (eps.max() - eps.min())), "SANC noise")]
    for a, (im, t) in zip(ax[0], top):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t)
    for a, t in zip(ax[1], steps):
        a.imshow(tensor_to_image(traj[t])); a.set_title(f"t = {t}")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout(); plt.savefig(out / f"forward_diffusion_{args.dataset}.png", dpi=110)
    print("Saved", out / f"forward_diffusion_{args.dataset}.png")


if __name__ == "__main__":
    main()
