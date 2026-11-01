"""Pipeline 05 - Reverse sampling with the trained model (cells 238, 256-272, 332).

  --sampler ddpm      ancestral DDPM from pure noise, posterior variance (v2)
  --sampler ddpm_beta first sampler (variance = beta_t)
  --sampler ddpm_input  DDPM started from the low-light image instead of noise
  --sampler ddim      Member-3 deterministic DDIM, t0=99, 25 steps, start from low-light  (used in the ablation)

Prints output statistics + PSNR/SSIM vs the normal-light reference and saves a figure.
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
from sanc.diffusion import ddim_sample, ddpm_sample  # noqa: E402
from sanc.metrics import psnr_batch, ssim_batch  # noqa: E402
from sanc.runtime import load_components  # noqa: E402
from sanc.utils import ensure_dir, set_seed, tensor_to_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm", "ddpm_beta", "ddpm_input"])
    ap.add_argument("--t0", type=int, default=C.ABLATION_T0)
    ap.add_argument("--steps", type=int, default=C.ABLATION_STEPS)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "05_diffusion"))
    args = ap.parse_args()
    set_seed(C.SEED)
    rt = load_components()
    out = ensure_dir(args.out)

    b = next(iter(rt.val_loader))
    low, ref = b["low_light"][: args.n].to(rt.device), b["original"][: args.n].to(rt.device)
    if args.sampler == "ddim":
        with torch.no_grad():
            M = rt.saliency(low)
        g = torch.Generator(device=rt.device).manual_seed(C.SEED)
        y = ddim_sample(rt.reverse, low, M, "sanc", rt.sanc_train, rt.schedule, args.t0, args.steps, g)
    else:
        y = ddpm_sample(low, rt.reverse, rt.saliency, rt.schedule,
                        variance="beta" if args.sampler == "ddpm_beta" else "posterior",
                        start_from_input=args.sampler == "ddpm_input")

    print(f"Output | mean {y.mean():.4f} | std {y.std():.4f} | range [{y.min():.3f}, {y.max():.3f}] | "
          f"finite {torch.isfinite(y).all().item()} | outside [0,1]: {((y < 0) | (y > 1)).float().mean():.3f}")
    print(f"Target mean {ref.mean():.4f} | low-light mean {low.mean():.4f}")
    print(f"PSNR low-light {psnr_batch(low, ref).mean():.3f} dB -> output {psnr_batch(y, ref).mean():.3f} dB")
    print(f"SSIM low-light {ssim_batch(low, ref).mean():.4f} -> output {ssim_batch(y, ref).mean():.4f}")

    k = min(4, low.shape[0])
    fig, ax = plt.subplots(k, 3, figsize=(9, 3 * k), squeeze=False)
    for i in range(k):
        for a, im, t in zip(ax[i], [low[i], y[i], ref[i]], ["Low-light", f"SANC ({args.sampler})", "Reference"]):
            a.imshow(tensor_to_image(im)); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(out / f"samples_{args.sampler}.png", dpi=110)
    print("Saved", out / f"samples_{args.sampler}.png")


if __name__ == "__main__":
    main()
