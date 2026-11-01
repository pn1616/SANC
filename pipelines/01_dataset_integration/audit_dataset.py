"""Pipeline 01 - Dataset audit.

Checks performed (consolidated from notebook cells 6, 52, 71, 91, 118, 142-143, 243, 247, 251-252):
  * duplicate (dataset, filename) rows
  * every original / low-light / mask file exists and is non-empty
  * image / mask sizes and modes, mask value encoding per dataset
    (Kvasir masks are 0..255, ORIGA/HAM masks are 0..1 -> must binarise with ``> 0``)
  * reports the known corrupt Kvasir mask (cju89y9h0puti0818i5yw29e6.jpg, 0 bytes)

Writes audit CSVs to results/01_dataset_integration/.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data.metadata import add_existence_flags, build_combined_df  # noqa: E402
from sanc.utils import ensure_dir  # noqa: E402


def mask_encoding(df, n=20, seed=0):
    rows = []
    for name, sub in df[df["complete"]].groupby("dataset"):
        for _, r in sub.sample(min(n, len(sub)), random_state=seed).iterrows():
            with Image.open(r["mask"]) as m:
                arr = np.array(m.convert("L"))
            with Image.open(r["low_light"]) as im:
                size, mode = im.size, im.mode
            rows.append(dict(dataset=name, filename=r["filename"], image_size=str(size), image_mode=mode,
                             mask_min=int(arr.min()), mask_max=int(arr.max()),
                             n_unique=len(np.unique(arr)), fg_ratio_gt0=float((arr > 0).mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "01_dataset_integration"))
    args = ap.parse_args()
    out = ensure_dir(args.out)

    df = add_existence_flags(build_combined_df())
    dup = df.duplicated(subset=["dataset", "filename"]).sum()
    print("Total rows:", len(df), "| duplicate dataset/filename rows:", dup)

    summary = df.groupby("dataset")[["original_exists", "low_light_exists", "mask_exists", "complete"]].sum()
    summary["rows"] = df.groupby("dataset").size()
    print("\nFile existence (non-empty files):\n", summary)
    summary.to_csv(out / "file_existence_summary.csv")

    missing = []
    for col in ["original", "low_light", "mask"]:
        bad = df[~df[col + "_exists"]]
        for _, r in bad.iterrows():
            size = os.path.getsize(r[col]) if os.path.exists(r[col]) else None
            missing.append(dict(dataset=r["dataset"], split=r["split"], filename=r["filename"], kind=col,
                                path=r[col], status="empty" if size == 0 else "missing"))
    missing = pd.DataFrame(missing)
    missing.to_csv(out / "missing_or_empty_files.csv", index=False)
    print(f"\nMissing/empty files: {len(missing)}")
    if len(missing):
        print(missing.head(20).to_string(index=False))
    if (df["filename"] == C.CORRUPT_MASK_FILENAME).any():
        print(f"\nKnown corrupt Kvasir mask present: {C.CORRUPT_MASK_FILENAME} "
              "(excluded from all evaluation; test split 410 -> 409)")

    enc = mask_encoding(df)
    enc.to_csv(out / "mask_encoding_sample.csv", index=False)
    print("\nMask encoding per dataset (sample):")
    print(enc.groupby("dataset")[["mask_min", "mask_max", "n_unique", "fg_ratio_gt0"]].agg(["min", "max"]))
    print("\nSaved audit files to", out)


if __name__ == "__main__":
    main()
