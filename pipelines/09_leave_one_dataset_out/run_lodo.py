"""Pipeline 09 - Leave-one-dataset-out (LODO) generalisation.

GENERATED: this notebook contains no LODO code, so this script implements the
protocol with the same components and hyper-parameters as the main pipeline.
If your team's LODO runs used a different protocol, replace this file.

For each held-out dataset H in {Kvasir-SEG, ORIGA, HAM10000}:
  1. saliency  - train U-Net (Experiment A recipe) on the train split of the
                 other two datasets. Validation = their 'val' rows; when Kvasir
                 is held out (ORIGA/HAM have no val split) 15 % of the training
                 pool is carved out, stratified by dataset (seed 42).
  2. reverse   - train DualBranchReverseUNet with SANC noise, conditioned on the
                 LODO saliency model, on the same training pool.
  3. evaluate  - on H's TEST split only (never seen):
                   * saliency Dice / IoU (per image, threshold 0.5)
                   * enhancement PSNR / SSIM for the low-light input, the SANC
                     baseline arm and the +A+B+C arm (or all 8 with --all-arms)
                     using the frozen ablation sampler (DDIM t0=99, 25 steps)

Outputs: $SANC_ROOT/models/lodo/<H>/{saliency,reverse}.pth and
         results/09_leave_one_dataset_out/<H>/*.csv + lodo_summary.csv

Usage:
    python pipelines/09_leave_one_dataset_out/run_lodo.py                    # all 3 folds, all stages
    python pipelines/09_leave_one_dataset_out/run_lodo.py --holdout ORIGA --stages evaluate
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import build_combined_df, get_split, make_loader  # noqa: E402
from sanc.diffusion import DiffusionSchedule, ddim_sample  # noqa: E402
from sanc.losses import BCEDiceLoss  # noqa: E402
from sanc.metrics import psnr_batch, ssim_batch  # noqa: E402
from sanc.models import UNet, DualBranchReverseUNet, load_reverse_model, load_saliency_model  # noqa: E402
from sanc.noise import ARMS, ArmNoise, ExtendedSANCNoiseColoring, SANCNoiseColoring, arm_name, compute_extended_saliency  # noqa: E402
from sanc.training import per_dataset_test_metrics, train_reverse_model, train_segmentation  # noqa: E402
from sanc.utils import ensure_dir, get_device, set_seed  # noqa: E402


def fold_splits(df, holdout):
    others = [d for d in C.DATASET_NAMES if d != holdout]
    train = get_split(df, "train", datasets=others)
    val = get_split(df, "val", datasets=others)
    if len(val) == 0:
        train, val = train_test_split(train, test_size=0.15, random_state=C.SEED, stratify=train["dataset"])
        train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    test = get_split(df, "test", datasets=[holdout])
    return train, val, test


def stage_saliency(train, val, ckpt, device, epochs, bs, log):
    set_seed(C.SEED)
    model = UNet(3, 1).to(device)
    hist, best = train_segmentation(model, BCEDiceLoss().to(device), torch.optim.Adam(model.parameters(), lr=1e-4),
                                    make_loader(train, bs, shuffle=True), make_loader(val, bs), device, epochs, ckpt, log=log)
    return hist


def stage_reverse(train, sal_ckpt, rev_ckpt, device, epochs, bs, log):
    set_seed(C.SEED)
    saliency = load_saliency_model(sal_ckpt, device)
    return train_reverse_model(DualBranchReverseUNet().to(device), saliency, SANCNoiseColoring(norm_eps=1e-8).to(device),
                               DiffusionSchedule(device=device), make_loader(train, bs, shuffle=True), device, epochs,
                               C.REVERSE_LR, C.REVERSE_WEIGHT_DECAY, save_path=rev_ckpt, log=log)


@torch.no_grad()
def stage_evaluate(test, sal_ckpt, rev_ckpt, device, arms, seed, bs):
    seg = load_saliency_model(sal_ckpt, device, return_probs=False, verbose=False)
    per_img_seg, seg_summary = per_dataset_test_metrics(seg, make_loader(test, bs), device, logits=True)

    saliency = load_saliency_model(sal_ckpt, device, verbose=False)
    reverse = load_reverse_model(rev_ckpt, device, verbose=False)
    sched = DiffusionSchedule(device=device)
    sanc_t = SANCNoiseColoring(norm_eps=1e-8).to(device)
    ext = ExtendedSANCNoiseColoring(SANCNoiseColoring(norm_eps=1e-6).to(device)).to(device)
    rows = []
    for bi, b in enumerate(make_loader(test, batch_size=32)):
        ll, ref = b["low_light"].to(device), b["original"].to(device)
        sal = compute_extended_saliency(ll, saliency)
        outs = {"Low-light input": ll}
        for arm in arms:
            g = torch.Generator(device=device).manual_seed(seed * 1000 + bi)
            outs[arm_name(*arm)] = ddim_sample(reverse, ll, sal["M"], "sanc", ArmNoise(sal, C.ABLATION_T0, arm, sanc_t, ext),
                                               sched, C.ABLATION_T0, C.ABLATION_STEPS, g).clamp(0, 1)
        for name, y in outs.items():
            p, s = psnr_batch(y, ref), ssim_batch(y, ref)
            rows += [dict(arm=name, filename=b["filename"][k], PSNR=p[k].item(), SSIM=s[k].item()) for k in range(ll.shape[0])]
    return per_img_seg, seg_summary, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", nargs="+", default=C.DATASET_NAMES, choices=C.DATASET_NAMES)
    ap.add_argument("--stages", nargs="+", default=["saliency", "reverse", "evaluate"],
                    choices=["saliency", "reverse", "evaluate"])
    ap.add_argument("--saliency-epochs", type=int, default=20)
    ap.add_argument("--reverse-epochs", type=int, default=C.REVERSE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default=str(C.MODELS_DIR / "lodo"))
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "09_leave_one_dataset_out"))
    args = ap.parse_args()
    device = get_device()
    df = build_combined_df()
    arms = ARMS if args.all_arms else [(0, 0, 0), (1, 1, 1)]
    summary = []

    for h in args.holdout:
        tag = h.replace("-", "")
        mdir, odir = ensure_dir(Path(args.models_dir) / tag), ensure_dir(Path(args.out) / tag)
        sal_ckpt, rev_ckpt = mdir / "saliency.pth", mdir / "reverse.pth"
        train, val, test = fold_splits(df, h)
        print(f"\n{'=' * 70}\nHOLD-OUT {h} | train {len(train)} {train['dataset'].value_counts().to_dict()} | "
              f"val {len(val)} | test {len(test)}\n{'=' * 70}")

        if "saliency" in args.stages:
            stage_saliency(train, val, sal_ckpt, device, args.saliency_epochs, args.batch_size, print) \
                .to_csv(odir / "saliency_history.csv", index=False)
        if "reverse" in args.stages:
            stage_reverse(train, sal_ckpt, rev_ckpt, device, args.reverse_epochs, args.batch_size, print) \
                .to_csv(odir / "reverse_history.csv", index=False)
        if "evaluate" in args.stages:
            seg_img, seg_sum, enh = stage_evaluate(test, sal_ckpt, rev_ckpt, device, arms, args.seed, args.batch_size)
            seg_img.to_csv(odir / "saliency_test_per_image.csv", index=False)
            enh.to_csv(odir / f"enhancement_test_seed{args.seed}_per_image.csv", index=False)
            means = enh.groupby("arm")[["PSNR", "SSIM"]].mean()
            print(seg_sum.round(4).to_string(index=False)); print(means.round(4))
            row = dict(holdout=h, n_test=len(test), saliency_dice=seg_img["dice"].mean(), saliency_iou=seg_img["iou"].mean())
            for a, r in means.iterrows():
                row[f"PSNR[{a}]"], row[f"SSIM[{a}]"] = r["PSNR"], r["SSIM"]
            summary.append(row)

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(ensure_dir(args.out) / "lodo_summary.csv", index=False)
        print("\nLODO summary\n", s.round(4).T.to_string())


if __name__ == "__main__":
    main()
