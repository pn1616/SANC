"""Pipeline 02 - Low-light generation.

Applies ``simulate_low_light`` (brightness x0.40, gamma 1.5, Gaussian noise
sigma 0.01, seed 42) to every image of a folder.

Lesson from the notebook (cells 41-53): writing thousands of files directly
into a mounted Google Drive folder silently dropped files. The default mode
therefore writes to a LOCAL folder first, verifies the count, then (optionally)
zips the folder and copies the zip to Drive.

Usage (ORIGA, as in the notebook):
    python pipelines/02_low_light_preprocessing/generate_low_light.py \
        --input  "$SANC_ROOT/datasets/ORIGA_processed/integration/images" \
        --output /content/ORIGA_low_light \
        --zip-to "$SANC_ROOT/datasets/ORIGA_processed/integration/ORIGA_low_light.zip"
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.preprocessing import simulate_low_light  # noqa: E402

EXT = (".png", ".jpg", ".jpeg")


def image_files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--brightness", type=float, default=C.LOW_LIGHT_BRIGHTNESS)
    ap.add_argument("--gamma", type=float, default=C.LOW_LIGHT_GAMMA)
    ap.add_argument("--noise-std", type=float, default=C.LOW_LIGHT_NOISE_STD)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--zip-to", default=None, help="optional: zip output and copy zip here")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    files = image_files(args.input)
    for i, f in enumerate(files, 1):
        img = np.array(Image.open(os.path.join(args.input, f)).convert("RGB"))
        low = simulate_low_light(img, args.brightness, args.gamma, args.noise_std)
        Image.fromarray(low).save(os.path.join(args.output, f))
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)}")

    made = image_files(args.output)
    missing = sorted(set(files) - set(made))
    print(f"Original: {len(files)} | generated: {len(made)} | missing: {len(missing)}")
    if missing:
        print("First missing:", missing[:20])
        sys.exit(1)

    if args.zip_to:
        base = os.path.splitext(args.output.rstrip("/"))[0]
        zip_file = shutil.make_archive(base, "zip", args.output)
        os.makedirs(os.path.dirname(args.zip_to), exist_ok=True)
        shutil.copy2(zip_file, args.zip_to)
        print("ZIP saved to", args.zip_to)


if __name__ == "__main__":
    main()
