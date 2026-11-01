"""Pipeline 01 - Dataset integration.

Builds the common metadata for Kvasir-SEG, ORIGA and HAM10000:
  1. (optional) create the Kvasir 70/15/15 split CSVs  (notebook cells 12-17)
  2. (optional) convert ORIGA_metadata.csv -> ORIGA_common_metadata.csv (cell 48)
  3. merge all three into one dataframe with the common schema (cell 243)
  4. save combined_metadata.csv + split summary

Usage:
    python pipelines/01_dataset_integration/build_metadata.py [--make-kvasir-splits] [--make-origa-common]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import pandas as pd  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data.metadata import build_combined_df, build_origa_common_metadata, make_kvasir_splits  # noqa: E402
from sanc.utils import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-kvasir-splits", action="store_true", help="re-create kvasir_{train,val,test}.csv")
    ap.add_argument("--make-origa-common", action="store_true", help="re-create ORIGA_common_metadata.csv")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "01_dataset_integration"))
    args = ap.parse_args()
    out = ensure_dir(args.out)

    if args.make_kvasir_splits:
        tr, va, te = make_kvasir_splits()
        print(f"Kvasir split written -> train {len(tr)} | val {len(va)} | test {len(te)}")
    if args.make_origa_common:
        o = build_origa_common_metadata()
        print(f"ORIGA common metadata written -> {len(o)} rows")

    df = build_combined_df()
    df.to_csv(out / "combined_metadata.csv", index=False)

    print("Total samples:", len(df))
    print("\nSamples per dataset:\n", df["dataset"].value_counts())
    print("\nSamples per split:\n", df["split"].value_counts())
    table = pd.crosstab(df["dataset"], df["split"])
    print("\nDataset x split:\n", table)
    table.to_csv(out / "dataset_split_counts.csv")
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
