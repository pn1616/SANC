"""Pipeline 08 - Eight-arm factorial ablation of SANC extensions A, B, C.

Sampler frozen to Member 3's deterministic DDIM (t0 = 99, 25 steps, start from
the low-light image). The same z is used for every arm (generator seed
seed*1000 + batch_index), so arm differences are paired.

Modes
  smoke   8 Kvasir validation images, all arms, 1 seed                        (cell 332)
  t0      seed noise floor + t0 sensitivity on 32 Kvasir val images           (cell 333)
  quick   8 test images per dataset, timing estimate                          (cell 334)
  full    all 409 test images x 8 arms x seeds -> per-image CSV               (cells 335-336)

Reported full run (seed 42, 409 images): baseline 12.446 dB -> +A+B+C 12.457 dB
(+0.011 dB), low-light input 10.043 dB. See results/08_ablation/.

Usage:
    python pipelines/08_ablation/run_ablation.py full --seeds 42
    python pipelines/08_ablation/run_ablation.py full --seeds 42 43 44 --tag 3seeds
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import make_loader  # noqa: E402
from sanc.diffusion import ddim_sample  # noqa: E402
from sanc.metrics import psnr_batch, ssim_batch  # noqa: E402
from sanc.noise import ARMS, ArmNoise, arm_name, compute_extended_saliency  # noqa: E402
from sanc.runtime import load_components  # noqa: E402
from sanc.utils import ensure_dir, set_seed  # noqa: E402

ORDER = ["Low-light input"] + [arm_name(*a) for a in ARMS]


def sample_arm(rt, ll, sal, arm, t0, steps, gen):
    noise = ArmNoise(sal, t0, arm, rt.sanc_train, rt.extended)
    return ddim_sample(rt.reverse, ll, sal["M"], "sanc", noise, rt.schedule, t0, steps, gen)


@torch.no_grad()
def run_ablation(rt, df, seeds, arms=ARMS, batch_size=32, t0=C.ABLATION_T0, steps=C.ABLATION_STEPS):
    rows = []
    for bi, b in enumerate(make_loader(df, batch_size=batch_size)):
        ll, ref = b["low_light"].to(rt.device), b["original"].to(rt.device)
        sal = compute_extended_saliency(ll, rt.saliency)
        p, s = psnr_batch(ll, ref), ssim_batch(ll, ref)
        for k in range(ll.shape[0]):
            rows.append(dict(arm="Low-light input", seed=-1, dataset=b["dataset"][k], filename=b["filename"][k],
                             PSNR=p[k].item(), SSIM=s[k].item()))
        for seed in seeds:
            for arm in arms:
                g = torch.Generator(device=rt.device).manual_seed(seed * 1000 + bi)
                out = sample_arm(rt, ll, sal, arm, t0, steps, g).clamp(0, 1)
                p, s = psnr_batch(out, ref), ssim_batch(out, ref)
                for k in range(ll.shape[0]):
                    rows.append(dict(arm=arm_name(*arm), seed=seed, dataset=b["dataset"][k], filename=b["filename"][k],
                                     PSNR=p[k].item(), SSIM=s[k].item()))
        print(f"  batch {bi + 1} done")
    return pd.DataFrame(rows)


def summarize(df):
    print(df.groupby("arm")[["PSNR", "SSIM"]].mean().reindex(ORDER).round(4), "\n")
    print("PSNR by dataset:")
    print(df.pivot_table(index="arm", columns="dataset", values="PSNR").reindex(ORDER).round(3))


@torch.no_grad()
def mode_t0(rt, out):
    kv = rt.val_df[rt.val_df["dataset"] == "Kvasir-SEG"]
    sub = kv.sample(n=min(32, len(kv)), random_state=0)
    b = next(iter(make_loader(sub, batch_size=32)))
    ll, x0 = b["low_light"].to(rt.device), b["original"].to(rt.device)
    sal = compute_extended_saliency(ll, rt.saliency)
    rows = []
    for t0 in [99, 75, 50, 25, 10]:
        pb, pf = [], []
        for s in [42, 43, 44]:
            for arm, store in [((0, 0, 0), pb), ((1, 1, 1), pf)]:
                g = torch.Generator(device=rt.device).manual_seed(s)
                store.append(psnr_batch(sample_arm(rt, ll, sal, arm, t0, min(t0 + 1, 25), g), x0))
        pb, pf = torch.stack(pb), torch.stack(pf)
        d = pf - pb
        rows.append(dict(t0=t0, base_psnr=pb.mean(1).mean().item(), seed_sd=pb.mean(1).std().item(),
                         full_minus_base=d.mean().item(), per_image_sd=d.std().item(), max_abs_diff=d.abs().max().item()))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "t0_sensitivity_val.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "t0", "quick", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "08_ablation"))
    args = ap.parse_args()
    set_seed(C.SEED)
    rt = load_components()
    out = ensure_dir(args.out)

    if args.mode == "t0":
        return mode_t0(rt, out)
    if args.mode == "smoke":
        df = rt.val_df[rt.val_df["dataset"] == "Kvasir-SEG"].head(8)
    elif args.mode == "quick":
        df = rt.test_df.groupby("dataset").head(8).reset_index(drop=True)
    else:
        df = rt.test_df
    print(f"{args.mode}: {len(df)} images | {df['dataset'].value_counts().to_dict()} | seeds {args.seeds}")

    start = time.time()
    res = run_ablation(rt, df, args.seeds, batch_size=args.batch_size)
    print(f"Done in {(time.time() - start) / 60:.1f} min | rows {len(res)}")
    summarize(res)
    tag = args.tag or f"{args.mode}_seed{'-'.join(map(str, args.seeds))}"
    path = out / f"ablation_{tag}_per_image.csv"
    if path.exists():
        path = path.with_name(path.stem + f"_{int(time.time())}.csv")   # never overwrite
    res.to_csv(path, index=False)
    print("Saved", path)


if __name__ == "__main__":
    main()
