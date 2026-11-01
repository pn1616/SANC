"""Pipeline 02 - Resize images to 128x128 and binarise masks to {0,255}.

GENERATED: the original resizing of Kvasir-SEG / HAM10000 was done before this
notebook (the processed folders on Drive are already 128x128). This script
reproduces that step so the repository covers preprocessing end-to-end.
Images use bilinear resampling; masks use nearest-neighbour and ``> 0``.

Usage:
    python pipelines/02_low_light_preprocessing/resize_and_binarize.py \
        --images raw/images --masks raw/masks --out processed/ [--size 128]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from PIL import Image  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.preprocessing import resize_and_binarize_mask, resize_image  # noqa: E402

EXT = (".png", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=C.IMAGE_SIZE)
    ap.add_argument("--image-dir-name", default="original")
    ap.add_argument("--mask-dir-name", default="masks")
    args = ap.parse_args()

    img_out = Path(args.out) / args.image_dir_name
    mask_out = Path(args.out) / args.mask_dir_name
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    n, skipped = 0, []
    for f in sorted(os.listdir(args.images)):
        if not f.lower().endswith(EXT):
            continue
        stem = os.path.splitext(f)[0]
        cands = [m for m in os.listdir(args.masks) if os.path.splitext(m)[0].startswith(stem)]
        if not cands:
            print("No mask for", f)
            continue
        try:
            img = Image.open(os.path.join(args.images, f)).convert("RGB")
            mask = Image.open(os.path.join(args.masks, cands[0]))
        except Exception as e:  # e.g. the 0-byte Kvasir mask
            skipped.append(f)
            print(f"Skipped {f}: {type(e).__name__}")
            continue
        resize_image(img, args.size).save(img_out / f)
        resize_and_binarize_mask(mask, args.size).save(mask_out / f)
        n += 1
    print(f"Wrote {n} image/mask pairs to {args.out} | skipped {len(skipped)}")


if __name__ == "__main__":
    main()
