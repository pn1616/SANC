"""Pipeline 07 - Extensions A, B, C: implementation checks and figures.

  maps        M, Sobel E, S (A), U, S_u (A+B) for one validation batch + figure  (cells 312-317, 328)
  timestep    Component C: band-energy shares at t = 0 / 50 / 99                 (cells 318, 321-324)
              reported: high-frequency share 0.244 -> 0.222 -> 0.202
  arms        all 8 ablation arms at t=99: mean/std/band shares/RMS vs baseline   (cell 329)
  regions     mean colouring map inside mask / edges / uncertain pixels,
              and frozen-model MSE ratio full vs baseline, per dataset            (cells 330-331)
  prototype   first UA-TD-SANC prototype (saliency-modulating) for the report     (cells 297-300)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import make_loader  # noqa: E402
from sanc.noise import ARMS, ablation_noise, arm_name, compute_extended_saliency  # noqa: E402
from sanc.noise.extensions import uncertainty_aware_sanc  # noqa: E402
from sanc.noise.sanc import band_energy_shares  # noqa: E402
from sanc.runtime import load_components  # noqa: E402
from sanc.utils import ensure_dir, set_seed, tensor_to_image  # noqa: E402


def first_val_batch(rt):
    b = next(iter(rt.val_loader))
    return {k: (v.to(rt.device) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def cmd_maps(rt, out):
    b = first_val_batch(rt)
    sal = compute_extended_saliency(b["low_light"], rt.saliency)
    for k, v in sal.items():
        print(f"{k:4s} mean {v.mean():.5f} | range [{v.min():.4f}, {v.max():.4f}] | finite {torch.isfinite(v).all().item()}")
    panels = [(tensor_to_image(b["low_light"][0]), "Low-light", None)] + \
             [(sal[k][0, 0].cpu(), t, c) for k, t, c in [("M", "U-Net saliency M", "hot"), ("E", "Multiscale Sobel E", "gray"),
                                                       ("S", "A: S", "hot"), ("U", "Entropy U", "viridis"), ("S_u", "A+B: S_u", "hot")]]
    fig, ax = plt.subplots(1, 6, figsize=(24, 4.5))
    for a, (im, t, c) in zip(ax, panels):
        a.imshow(im, cmap=c, vmin=0, vmax=1); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(out / "extension_maps.png", dpi=110)
    pd.DataFrame([{k: v.mean().item() for k, v in sal.items()}]).to_csv(out / "extension_map_means.csv", index=False)


@torch.no_grad()
def cmd_timestep(rt, out):
    b = first_val_batch(rt)
    S_u = compute_extended_saliency(b["low_light"], rt.saliency)["S_u"]
    torch.manual_seed(42)
    z = torch.randn(S_u.shape[0], 3, *S_u.shape[-2:], device=rt.device)
    rows = []
    for t in (0, 50, 99):
        y = rt.extended(z, S_u, t, C.T)
        lo, mi, hi = band_energy_shares(y)
        rows.append(dict(t=t, low=lo, mid=mi, high=hi, mean=y.mean().item(), std=y.std().item()))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "component_C_band_shares.csv", index=False)


@torch.no_grad()
def cmd_arms(rt, out):
    b = first_val_batch(rt)
    sal = compute_extended_saliency(b["low_light"], rt.saliency)
    g = torch.Generator(device=rt.device).manual_seed(0)
    z = torch.randn(b["low_light"].shape, generator=g, device=rt.device)
    eps = {arm_name(*a): ablation_noise(z, sal, C.T - 1, *a, sanc=rt.sanc_train, extended_sanc=rt.extended) for a in ARMS}
    base = eps["SANC (baseline)"]
    rows = []
    for n, x in eps.items():
        lo, mi, hi = band_energy_shares(x)
        rows.append(dict(arm=n, mean=x.mean().item(), std=x.std().item(), low=lo, mid=mi, high=hi,
                         rms_vs_baseline=(x - base).pow(2).mean().sqrt().item()))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "ablation_arms_noise_stats_t99.csv", index=False)


@torch.no_grad()
def cmd_regions(rt, out, n=16):
    rows = []
    for ds in C.DATASET_NAMES:
        pool = rt.train_df[rt.train_df["dataset"] == ds]
        sub = pool.sample(n=min(n, len(pool)), random_state=0)
        b = next(iter(make_loader(sub, batch_size=len(sub))))
        ll, x0, mask = (b[k].to(rt.device) for k in ("low_light", "original", "mask"))
        sal = compute_extended_saliency(ll, rt.saliency)
        for t in (10, 25, 50, 99):
            tt = torch.full((len(sub),), t, device=rt.device, dtype=torch.long)
            mse = {}
            for arm in [(0, 0, 0), (1, 1, 1)]:
                vals = []
                for s in (0, 1):
                    z = torch.randn(x0.shape, generator=torch.Generator(device=rt.device).manual_seed(s), device=rt.device)
                    e = ablation_noise(z, sal, t, *arm, sanc=rt.sanc_train, extended_sanc=rt.extended)
                    vals.append(F.mse_loss(rt.reverse(rt.schedule.q_sample(x0, tt, e), ll, sal["M"], tt), e).item())
                mse[arm] = sum(vals) / len(vals)
            print(f"{ds:10s} t={t:2d} | MSE baseline {mse[(0, 0, 0)]:.5f} | A+B+C {mse[(1, 1, 1)]:.5f} | "
                  f"ratio {mse[(1, 1, 1)] / mse[(0, 0, 0)]:.2f}x")
        E = sal["E"]
        regions = {"inside_mask": mask > 0.5, "edges_top10": E >= torch.quantile(E.flatten()[::4].float(), 0.90),
                   "uncertain": sal["U"] > 0.5}
        maps = {"M": sal["M"], "A:S": sal["S"], "B:M(1-lU)": (sal["M"] * (1 - C.LAMBDA_U * sal["U"])).clamp(0, 1), "A+B:S_u": sal["S_u"]}
        for mn, m in maps.items():
            rows.append(dict(dataset=ds, map=mn, **{r: m[sel].mean().item() if sel.any() else float("nan")
                                                    for r, sel in regions.items()}))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "regional_behaviour.csv", index=False)


@torch.no_grad()
def cmd_prototype(rt, out):
    b = first_val_batch(rt)
    M = rt.saliency(b["low_light"])
    z = torch.randn_like(b["low_light"])
    for t in (0, 25, 50, 75, 99):
        y, eff, U = uncertainty_aware_sanc(z, M, t, C.T, rt.sanc_5a)
        print(f"t={t:2d} | effective saliency {eff.mean():.4f} | U {U.mean():.4f} | std {y.std():.4f} | finite {torch.isfinite(y).all().item()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["maps", "timestep", "arms", "regions", "prototype"])
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "07_extensions_abc"))
    args = ap.parse_args()
    set_seed(C.SEED)
    rt = load_components()
    {"maps": cmd_maps, "timestep": cmd_timestep, "arms": cmd_arms, "regions": cmd_regions,
     "prototype": cmd_prototype}[args.command](rt, ensure_dir(args.out))


if __name__ == "__main__":
    main()
