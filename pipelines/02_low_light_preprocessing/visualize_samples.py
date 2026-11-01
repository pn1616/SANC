"""Pipeline 02 - Visual check: original / low-light / mask for a few samples per dataset (cells 11, 78)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import SANCMedicalDataset, build_combined_df, get_split  # noqa: E402
from sanc.utils import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=2)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "02_low_light_preprocessing"))
    args = ap.parse_args()
    out = ensure_dir(args.out)

    df = get_split(build_combined_df(), "train")
    rows = []
    for name in C.DATASET_NAMES:
        sub = df[df["dataset"] == name].head(args.per_dataset)
        ds = SANCMedicalDataset(sub)
        rows += [ds[i] for i in range(len(ds))]

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    for i, s in enumerate(rows):
        print(f"{s['dataset']:10s} {s['filename']} | low-light mean {s['low_light'].mean():.3f} | "
              f"original mean {s['original'].mean():.3f} | mask fg {s['mask'].mean():.3f}")
        axes[i, 0].imshow(s["original"].permute(1, 2, 0)); axes[i, 0].set_title(f"{s['dataset']}\nOriginal")
        axes[i, 1].imshow(s["low_light"].permute(1, 2, 0)); axes[i, 1].set_title("Low-light")
        axes[i, 2].imshow(s["mask"][0], cmap="gray"); axes[i, 2].set_title("Mask")
        for a in axes[i]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(out / "low_light_samples.png", dpi=120)
    print("Saved", out / "low_light_samples.png")


if __name__ == "__main__":
    main()
