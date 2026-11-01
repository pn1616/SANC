"""Dataset integration: build one common dataframe for Kvasir-SEG, ORIGA and HAM10000.

Every dataset is reduced to the common schema
``filename, original, low_light, mask, dataset, split`` and its paths are
re-resolved against the persistent Drive folders (stale ``/content/...``
paths in older CSVs are ignored). Logic follows the consolidated cell
"SANC SALIENCY DEBUGGING - CONSOLIDATED DATASET SETUP" of the notebook.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config as C
from ..utils import is_valid_file


# ----------------------------------------------------------------------------
# Kvasir-SEG
# ----------------------------------------------------------------------------
def _filename_column(df):
    for col in ["filename", "image", "image_id", "id"]:
        if col in df.columns:
            return col
    raise ValueError(f"No filename column found. Columns: {list(df.columns)}")


def make_kvasir_splits(kvasir_meta_csv=C.KVASIR_META_CSV, out_dir=C.SPLIT_META_DIR,
                       seed=C.SEED, write=True):
    """70/15/15 split of Kvasir-SEG (random_state=42), as in notebook cell 12."""
    df = pd.read_csv(kvasir_meta_csv)
    df["filename"] = df[_filename_column(df)].astype(str).apply(os.path.basename)
    # Exact literals from the notebook (0.30 then 0.50) so the split is reproduced bit-for-bit
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)
    if write:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "kvasir_train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "kvasir_val.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "kvasir_test.csv"), index=False)
    return train_df, val_df, test_df


def load_kvasir_df(split_dir=C.SPLIT_META_DIR):
    parts = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(split_dir, f"kvasir_{split}.csv"))
        out = pd.DataFrame({"filename": df[_filename_column(df)].astype(str).apply(os.path.basename)})
        out["dataset"] = "Kvasir-SEG"
        out["split"] = split
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# ORIGA
# ----------------------------------------------------------------------------
def build_origa_common_metadata(raw_csv=C.ORIGA_RAW_META_CSV, out_csv=C.ORIGA_META_CSV, write=True):
    """ORIGA_metadata.csv -> ORIGA_common_metadata.csv (notebook cell 48)."""
    origa = pd.read_csv(raw_csv)
    common = pd.DataFrame({
        "filename": origa["image_path"].apply(os.path.basename),
        "original": origa["image_path"].apply(lambda x: str(C.ORIGA_ROOT / x)),
        "low_light": origa["image_path"].apply(lambda x: str(C.ORIGA_ROOT / "low_light" / os.path.basename(x))),
        "mask": origa["mask_path"].apply(lambda x: str(C.ORIGA_ROOT / x)),
        "dataset": origa["dataset_name"],
        "split": origa["split"],
    })
    if write:
        common.to_csv(out_csv, index=False)
    return common


def load_origa_df(meta_csv=C.ORIGA_META_CSV):
    df = pd.read_csv(meta_csv)
    out = pd.DataFrame({"filename": df["filename"].astype(str).apply(os.path.basename)})
    out["dataset"] = "ORIGA"
    out["split"] = df["split"].astype(str).str.strip().str.lower()
    return out


# ----------------------------------------------------------------------------
# HAM10000
# ----------------------------------------------------------------------------
def load_ham_df(meta_csv=C.HAM_META_CSV):
    df = pd.read_csv(meta_csv)
    out = pd.DataFrame({
        "filename": df["image_id"].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0] + ".png")
    })
    out["dataset"] = "HAM10000"
    out["split"] = df["split"].astype(str).str.strip().str.lower()
    return out


# ----------------------------------------------------------------------------
# Combined
# ----------------------------------------------------------------------------
def resolve_paths(df):
    """Fill original / low_light / mask from the persistent Drive layout."""
    df = df.copy()
    for name, layout in C.DATASET_LAYOUT.items():
        idx = df["dataset"].eq(name)
        for col in ["original", "low_light", "mask"]:
            folder = os.path.join(str(layout["root"]), layout[col])
            df.loc[idx, col] = df.loc[idx, "filename"].apply(lambda f, d=folder: os.path.join(d, f))
    return df


def build_combined_df(datasets=None):
    """Return the 2,300-row common dataframe (Kvasir 1000, ORIGA 650, HAM 650)."""
    loaders = {"Kvasir-SEG": load_kvasir_df, "ORIGA": load_origa_df, "HAM10000": load_ham_df}
    datasets = datasets or C.DATASET_NAMES
    df = pd.concat([loaders[d]() for d in datasets], ignore_index=True)
    df = resolve_paths(df)
    return df[C.COMMON_COLUMNS]


def add_existence_flags(df):
    df = df.copy()
    for col in ["original", "low_light", "mask"]:
        df[col + "_exists"] = df[col].map(is_valid_file)
    df["complete"] = df[["original_exists", "low_light_exists", "mask_exists"]].all(axis=1)
    return df


def exclude_corrupt(df):
    """Drop the Kvasir record with the empty mask file (410 -> 409 test images)."""
    return df[~df["filename"].astype(str).str.endswith(C.CORRUPT_MASK_FILENAME)].reset_index(drop=True)


def get_split(df, split, datasets=None, drop_corrupt=True):
    out = df[df["split"].astype(str).str.lower() == split]
    if datasets is not None:
        out = out[out["dataset"].isin(datasets)]
    if drop_corrupt:
        out = exclude_corrupt(out)
    return out.reset_index(drop=True)
