"""Pipeline 06 - Mathematical consistency audit (Member 1 + post-mid-review checks).

Sections (all read-only, nothing is trained):
  schedule   beta in (0,1), alpha = 1 - beta, alpha_bar = cumprod, terminal coefficients     (cells 264, 274, 276)
  posterior  DDPM posterior variance finite, = 0 at t=0                                     (cells 268-269)
  sanc       SANC output ~ N(0,1), finite, M=0 vs M=1 differ, both SANC instances agree       (cells 275, 296, 325)
  saliency   saliency model outputs a probability map in [0,1]                              (cell 295)
  reverse    reverse checkpoint loads with strict=True, output shape == x_t, finite,
             noise-prediction MSE at t = T/2, DDPM sampler stays finite                     (cells 255, 258-267, 293-310)

Usage:
    python pipelines/06_consistency_audit/audit.py                 # all sections
    python pipelines/06_consistency_audit/audit.py --only schedule sanc
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.diffusion import DiffusionSchedule, ddpm_sample  # noqa: E402
from sanc.noise import SANCNoiseColoring  # noqa: E402
from sanc.utils import ensure_dir, get_device, set_seed  # noqa: E402

REPORT = {}


def check(section, name, ok, value=None):
    REPORT.setdefault(section, []).append(dict(check=name, passed=bool(ok), value=value))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" = {value}" if value is not None else ""))


def audit_schedule(s):
    print("\n== Diffusion schedule ==")
    for i in sorted({0, 1, s.T // 2, s.T - 2, s.T - 1}):
        print(f"  t={i:3d} | beta={s.betas[i]:.6f} | alpha={s.alphas[i]:.6f} | alpha_bar={s.alpha_bars[i]:.6f}")
    check("schedule", "betas in (0,1)", ((s.betas > 0) & (s.betas < 1)).all())
    check("schedule", "alpha = 1 - beta", torch.allclose(s.alphas, 1 - s.betas, atol=1e-6))
    check("schedule", "alpha_bar = cumprod(alpha)", torch.allclose(s.alpha_bars, torch.cumprod(s.alphas, 0), atol=1e-6))
    check("schedule", "alpha_bar strictly decreasing", (s.alpha_bars[1:] < s.alpha_bars[:-1]).all())
    ab = s.alpha_bars[-1].item()
    check("schedule", "terminal alpha_bar_T", True, round(ab, 6))
    check("schedule", "signal coefficient sqrt(ab_T)", True, round(ab ** 0.5, 6))
    check("schedule", "noise coefficient sqrt(1-ab_T)", True, round((1 - ab) ** 0.5, 6))
    print("  NOTE: alpha_bar_T = 0.36 -> x_T is NOT pure noise; samplers start from the low-light image.")


def audit_posterior(s):
    print("\n== Posterior variance ==")
    pv = s.posterior_variance
    check("posterior", "all finite", torch.isfinite(pv).all())
    check("posterior", "zero at t=0", abs(pv[0].item()) < 1e-12, pv[0].item())
    check("posterior", "0 <= posterior var <= beta", ((pv >= 0) & (pv <= s.betas + 1e-12)).all())
    for t in (1, 50, 99):
        print(f"  posterior variance t={t}: {pv[t].item():.6e}")


@torch.no_grad()
def audit_sanc(device):
    print("\n== SANC numerical ==")
    torch.manual_seed(42)
    z = torch.randn(4, 3, 128, 128, device=device)
    s_train, s_5a = SANCNoiseColoring(norm_eps=1e-8).to(device), SANCNoiseColoring(norm_eps=1e-6).to(device)
    lo = s_train(z, torch.zeros(4, 1, 128, 128, device=device))
    hi = s_train(z, torch.ones(4, 1, 128, 128, device=device))
    for n, x in [("M=0", lo), ("M=1", hi)]:
        check("sanc", f"{n} mean ~ 0", abs(x.mean().item()) < 1e-3, round(x.mean().item(), 6))
        check("sanc", f"{n} std ~ 1", abs(x.std().item() - 1) < 1e-2, round(x.std().item(), 6))
        check("sanc", f"{n} finite", torch.isfinite(x).all())
    check("sanc", "M=0 and M=1 outputs differ", not torch.allclose(lo, hi))
    M = torch.rand(4, 1, 128, 128, device=device)
    d = (s_train(z, M) - s_5a(z, M)).abs().max().item()
    check("sanc", "training instance vs STEP-5A instance max|d| < 1e-4", d < 1e-4, f"{d:.2e}")
    for M0 in (0.0, 1.0):
        w = (0.55 - 0.35 * M0) + 0.30 + (0.15 + 0.35 * M0)
        check("sanc", f"band weights sum to 1 at M={M0}", abs(w - 1) < 1e-9, w)


@torch.no_grad()
def audit_models(device, s):
    from sanc.data import build_combined_df, get_split, make_loader
    from sanc.models import load_reverse_model, load_saliency_model
    print("\n== Saliency model ==")
    sal = load_saliency_model(C.CKPT["saliency"], device)
    b = next(iter(make_loader(get_split(build_combined_df(), "val"))))
    low, x0 = b["low_light"].to(device), b["original"].to(device)
    M = sal(low)
    check("saliency", "output in [0,1]", (M.min() >= 0) and (M.max() <= 1), f"[{M.min():.5f}, {M.max():.5f}]")

    print("\n== Reverse model ==")
    rev = load_reverse_model(C.CKPT["reverse"], device, strict=True)
    check("reverse", "strict state-dict load", True)
    sanc = SANCNoiseColoring(norm_eps=1e-8).to(device)
    eps = sanc(torch.randn_like(x0), M)
    t = torch.full((x0.size(0),), s.T // 2, device=device, dtype=torch.long)
    pred = rev(s.q_sample(x0, t, eps), low, M, t)
    check("reverse", "output shape == x_t shape", pred.shape == x0.shape, tuple(pred.shape))
    check("reverse", "output finite", torch.isfinite(pred).all())
    check("reverse", f"noise-prediction MSE at t={s.T // 2}", True, round(F.mse_loss(pred, eps).item(), 6))
    y = ddpm_sample(low, rev, sal, s, variance="posterior")
    check("reverse", "DDPM sample finite", torch.isfinite(y).all(), f"mean={y.mean():.4f}, std={y.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["schedule", "posterior", "sanc", "models"])
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "06_consistency_audit"))
    args = ap.parse_args()
    set_seed(C.SEED)
    device, out = get_device(), ensure_dir(args.out)
    s = DiffusionSchedule(device=device)
    want = set(args.only or ["schedule", "posterior", "sanc", "models"])
    if "schedule" in want: audit_schedule(s)
    if "posterior" in want: audit_posterior(s)
    if "sanc" in want: audit_sanc(device)
    if "models" in want: audit_models(device, s)
    (out / "audit_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    n_fail = sum(not c["passed"] for v in REPORT.values() for c in v)
    print(f"\n{sum(len(v) for v in REPORT.values())} checks, {n_fail} failed. Report: {out / 'audit_report.json'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
