"""Pipeline 07 - Extension audits A1, A2, A3 (read-only, no training).

A1  Noise-module equivalence: training SANC == STEP-5A SANC == Extended(t=0, M)
A2  Frozen-model compatibility: noise-prediction MSE of the frozen reverse model
    under each extended colouring vs original SANC (ratio ~ 0.97x-1.02x)
A3  Forward-chain validity of timestep-dependent colouring. For x_t to be a
    valid Markov chain, every increment covariance must be PSD:
        (1 - ab_t) P_t(f) - a_t (1 - ab_{t-1}) P_{t-1}(f) >= 0   for all f, t
    Result: lambda_t = 0.05 VALID, 0.10 INVALID -> lambda_t fixed at 0.05.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.noise import compute_extended_saliency  # noqa: E402
from sanc.noise.sanc import frequency_bands  # noqa: E402
from sanc.runtime import load_components  # noqa: E402
from sanc.utils import ensure_dir, set_seed  # noqa: E402


@torch.no_grad()
def a1(rt, low, sal):
    torch.manual_seed(0)
    z = torch.randn_like(low)
    M, S_u = sal["M"], sal["S_u"]
    ref = rt.sanc_train(z, M)
    variants = {"sanc (STEP 5A), M": rt.sanc_5a(z, M), "extended t=0, M": rt.extended(z, M, 0, C.T),
                "extended t=0, S_u": rt.extended(z, S_u, 0, C.T), "extended t=99, S_u": rt.extended(z, S_u, C.T - 1, C.T)}
    rows = [dict(variant=n, max_abs_diff=(e - ref).abs().max().item(), rms_diff=(e - ref).pow(2).mean().sqrt().item())
            for n, e in variants.items()]
    return pd.DataFrame(rows)


@torch.no_grad()
def a2(rt, low, x0, sal):
    M, S_u = sal["M"], sal["S_u"]
    B = x0.shape[0]
    names = ["Original SANC", "C only", "A+B only", "A+B+C"]
    rows = []
    for t in [0, 10, 25, 50, 75, 99]:
        tt = torch.full((B,), t, device=rt.device, dtype=torch.long)
        acc = {n: [] for n in names}; gap = {n: [] for n in names}
        for s in range(4):
            z = torch.randn(x0.shape, generator=torch.Generator(device=rt.device).manual_seed(s), device=rt.device)
            eps = {"Original SANC": rt.sanc_train(z, M), "C only": rt.extended(z, M, t, C.T),
                   "A+B only": rt.extended(z, S_u, 0, C.T), "A+B+C": rt.extended(z, S_u, t, C.T)}
            for n in names:
                pred = rt.reverse(rt.schedule.q_sample(x0, tt, eps[n]), low, M, tt)
                acc[n].append(F.mse_loss(pred, eps[n]).item())
                gap[n].append(F.mse_loss(eps[n], eps["Original SANC"]).item())
        base = sum(acc["Original SANC"]) / 4
        for n in names:
            m = sum(acc[n]) / 4
            rows.append(dict(t=t, variant=n, mse=m, ratio_vs_sanc=m / base, target_gap=sum(gap[n]) / 4))
    return pd.DataFrame(rows)


def a3(rt, S_u, lambdas=(0.0, 0.05, 0.10, 0.15, 0.50)):
    s = rt.schedule
    H, W = S_u.shape[-2:]
    lb, mb, hb = frequency_bands(H, W, rt.device)

    def P(S, tau, lam):
        wl = max(0.55 - 0.35 * S + lam * tau, 0.0); wm = 0.30; wh = max(0.15 + 0.35 * S - lam * tau, 0.0)
        p = ((wl * lb + wm * mb + wh * hb) / (wl + wm + wh)) ** 2
        return p / p.mean()

    q = torch.quantile(S_u.flatten().float()[::4], torch.tensor([0.5, 0.99], device=rt.device))
    levels = [0.0, q[0].item(), q[1].item(), 1.0]
    rows = []
    for lam in lambdas:
        worst, where = float("inf"), None
        for S in levels:
            for t in range(1, s.T):
                inc = (1 - s.alpha_bars[t]) * P(S, t / (s.T - 1), lam) - s.alphas[t] * (1 - s.alpha_bars[t - 1]) * P(S, (t - 1) / (s.T - 1), lam)
                v = (inc.min() / s.betas[t]).item()
                if v < worst:
                    worst, where = v, (S, t)
        rows.append(dict(lambda_t=lam, worst_increment_over_beta=worst, at_S=where[0], at_t=where[1],
                         result="VALID" if worst >= 0 else "INVALID"))
    return pd.DataFrame(rows), levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["A1", "A2", "A3"], default=None)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "07_extensions_abc"))
    args = ap.parse_args()
    set_seed(C.SEED)
    rt = load_components()
    out = ensure_dir(args.out)
    b = next(iter(rt.val_loader))
    low, x0 = b["low_light"].to(rt.device), b["original"].to(rt.device)
    sal = compute_extended_saliency(low, rt.saliency)
    want = set(args.only or ["A1", "A2", "A3"])
    if "A1" in want:
        d = a1(rt, low, sal); print("\nA1 equivalence vs training-time SANC\n", d.to_string(index=False))
        d.to_csv(out / "audit_A1_equivalence.csv", index=False)
    if "A2" in want:
        d = a2(rt, low, x0, sal)
        print("\nA2 frozen-model MSE ratio\n", d.pivot(index="t", columns="variant", values="ratio_vs_sanc").round(3))
        d.to_csv(out / "audit_A2_frozen_model.csv", index=False)
    if "A3" in want:
        d, lv = a3(rt, sal["S_u"]); print(f"\nA3 forward-chain validity (S levels {[round(x, 4) for x in lv]})\n", d.to_string(index=False))
        d.to_csv(out / "audit_A3_forward_chain.csv", index=False)


if __name__ == "__main__":
    main()
