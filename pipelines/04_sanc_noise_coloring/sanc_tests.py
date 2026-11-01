"""Pipeline 04 - SANC formulation and FFT noise-colouring tests.

  synthetic  numerical test on a synthetic smooth square saliency (cell 191)
             + spectrum figure (cell 192)
  spectral   all-low vs all-high saliency: low/high radial energy (cell 193)
             reported: HF/LF ratio 0.473 (M=0) vs 1.753 (M=1)
  real       real low-light image -> U-Net saliency -> SANC noise,
             local high-frequency energy inside high vs low saliency (cells 203-204)

Usage:
    python pipelines/04_sanc_noise_coloring/sanc_tests.py synthetic
    python pipelines/04_sanc_noise_coloring/sanc_tests.py spectral
    python pipelines/04_sanc_noise_coloring/sanc_tests.py real
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.noise import SANCNoiseColoring  # noqa: E402
from sanc.noise.sanc import local_high_frequency_energy, radial_energy  # noqa: E402
from sanc.utils import describe, ensure_dir, get_device, set_seed  # noqa: E402


@torch.no_grad()
def synthetic(sanc, device, out):
    noise = torch.randn(2, 3, 128, 128, device=device)
    sal = torch.zeros(2, 1, 128, 128, device=device)
    sal[:, :, 32:96, 32:96] = 1.0
    sal = F.avg_pool2d(sal, 15, stride=1, padding=7)
    colored = sanc(noise, sal)
    print(describe("Gaussian", noise)); print(describe("SANC", colored))
    for M in (0.0, 1.0):
        print(f"M={M}: weights low={0.55 - 0.35 * M:.2f} mid=0.30 high={0.15 + 0.35 * M:.2f}")
    g, s = noise[0, 0].cpu().numpy(), colored[0, 0].cpu().numpy()
    spec = lambda a: np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a))))  # noqa: E731
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    for a, im, t in zip(ax, [sal[0, 0].cpu(), g, spec(g), s, spec(s)],
                        ["Saliency map", "Gaussian noise", "Gaussian spectrum", "SANC noise", "SANC spectrum"]):
        a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(out / "sanc_synthetic_spectrum.png", dpi=110)
    return dict(sanc_mean=colored.mean().item(), sanc_std=colored.std().item(),
                nan=bool(torch.isnan(colored).any()), inf=bool(torch.isinf(colored).any()))


@torch.no_grad()
def spectral(sanc, device, out):
    noise = torch.randn(1, 3, 128, 128, device=device)
    lo = sanc(noise, torch.zeros(1, 1, 128, 128, device=device))
    hi = sanc(noise, torch.ones(1, 1, 128, 128, device=device))
    (ll, lh), (hl, hh) = radial_energy(lo[0, 0]), radial_energy(hi[0, 0])
    res = dict(low_saliency_LF=float(ll), low_saliency_HF=float(lh), high_saliency_LF=float(hl),
               high_saliency_HF=float(hh), ratio_low_saliency=float(lh / ll), ratio_high_saliency=float(hh / hl))
    for k, v in res.items():
        print(f"{k:22s}: {v:.4f}")
    print("PASS" if res["ratio_high_saliency"] > res["ratio_low_saliency"] else "FAIL",
          "- high saliency shifts energy to high frequencies")
    return res


@torch.no_grad()
def real(sanc, device, out):
    from sanc.data import SANCMedicalDataset, build_combined_df, get_split
    from sanc.models import load_saliency_model
    saliency = load_saliency_model(C.CKPT["saliency"], device, return_probs=True)
    df = get_split(build_combined_df(), "test")
    res = {}
    for name in C.DATASET_NAMES:
        s = SANCMedicalDataset(df[df["dataset"] == name].head(1))[0]
        low = s["low_light"].unsqueeze(0).to(device)
        M = saliency(low)
        N = sanc(torch.randn_like(low), M)[0].mean(0)
        hi_e = local_high_frequency_energy(N, M[0, 0] >= 0.70)
        lo_e = local_high_frequency_energy(N, M[0, 0] <= 0.30)
        res[name] = dict(filename=s["filename"], saliency_mean=M.mean().item(), hf_high_saliency=hi_e,
                         hf_low_saliency=lo_e, ratio=hi_e / (lo_e + 1e-8))
        print(f"{name:10s} | HF energy high-sal {hi_e:.4f} | low-sal {lo_e:.4f} | ratio {res[name]['ratio']:.3f} "
              f"| {'PASS' if res[name]['ratio'] > 1 else 'WARN'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["synthetic", "spectral", "real"])
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "04_sanc_noise_coloring"))
    args = ap.parse_args()
    set_seed(C.SEED)
    device, out = get_device(), ensure_dir(args.out)
    sanc = SANCNoiseColoring(norm_eps=1e-6).to(device)
    res = {"synthetic": synthetic, "spectral": spectral, "real": real}[args.test](sanc, device, out)
    (out / f"sanc_{args.test}_test.json").write_text(json.dumps(res, indent=2, default=float))
    print("Saved results to", out)


if __name__ == "__main__":
    main()
