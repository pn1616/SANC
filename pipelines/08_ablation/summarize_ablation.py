"""Pipeline 08 - Summary tables from a per-image ablation CSV.

Outputs (next to the input CSV):
  *_summary.csv          mean PSNR / SSIM per arm
  *_by_dataset.csv       mean PSNR per arm x dataset
  *_paired_vs_base.csv   paired difference vs SANC baseline per arm
                         (mean, SD, 95% CI, share of images improved, Wilcoxon p if scipy is available)

Usage:
    python pipelines/08_ablation/summarize_ablation.py results/08_ablation/ablation_test_seed42_per_image.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sanc.noise import ARMS, arm_name  # noqa: E402

ORDER = ["Low-light input"] + [arm_name(*a) for a in ARMS]


def paired(df, metric):
    arms = df[df["arm"] != "Low-light input"]
    wide = arms.pivot_table(index=["dataset", "filename", "seed"], columns="arm", values=metric)
    base = wide["SANC (baseline)"]
    rows = []
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for a in [arm_name(*x) for x in ARMS[1:]]:
        d = (wide[a] - base).dropna()
        se = d.std(ddof=1) / np.sqrt(len(d))
        p = wilcoxon(d).pvalue if (wilcoxon and (d != 0).any()) else np.nan
        rows.append(dict(metric=metric, arm=a, n=len(d), mean_diff=d.mean(), sd=d.std(ddof=1),
                         ci95_low=d.mean() - 1.96 * se, ci95_high=d.mean() + 1.96 * se,
                         share_improved=(d > 0).mean(), wilcoxon_p=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    p = Path(args.csv)
    df = pd.read_csv(p)
    s = df.groupby("arm")[["PSNR", "SSIM"]].mean().reindex(ORDER)
    s["PSNR_gain_vs_input"] = s["PSNR"] - s.loc["Low-light input", "PSNR"]
    by = df.pivot_table(index="arm", columns="dataset", values="PSNR").reindex(ORDER)
    pr = pd.concat([paired(df, "PSNR"), paired(df, "SSIM")], ignore_index=True)
    print(s.round(4), "\n\n", by.round(3), "\n\n", pr.round(5).to_string(index=False))
    s.to_csv(p.with_name(p.stem + "_summary.csv"))
    by.to_csv(p.with_name(p.stem + "_by_dataset.csv"))
    pr.to_csv(p.with_name(p.stem + "_paired_vs_base.csv"), index=False)


if __name__ == "__main__":
    main()
