"""Pipeline 05 - Train the conditional dual-branch reverse network (cells 207-237).

Per batch: M = U-Net(low-light) (frozen) -> eps = SANC(z, M) -> t ~ U{0..99}
-> x_t = q_sample(x0, t, eps) -> loss = MSE(eps_theta(x_t, low-light, M, t), eps).
AdamW lr 1e-4, wd 1e-4, 10 epochs, batch 8 (final train loss 0.0234).

Usage:
    python pipelines/05_diffusion/train_reverse.py [--epochs 10] [--sanity-only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import torch  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import build_combined_df, get_split, make_loader  # noqa: E402
from sanc.diffusion import DiffusionSchedule  # noqa: E402
from sanc.models import DualBranchReverseUNet, load_saliency_model  # noqa: E402
from sanc.noise import SANCNoiseColoring  # noqa: E402
from sanc.training import reverse_training_step, train_reverse_model  # noqa: E402
from sanc.utils import count_parameters, ensure_dir, get_device, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.REVERSE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--saliency-ckpt", default=str(C.CKPT["saliency"]))
    ap.add_argument("--save-path", default=str(C.CKPT["reverse"]))
    ap.add_argument("--sanity-only", action="store_true", help="one forward/loss check, no training")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "05_diffusion"))
    args = ap.parse_args()
    set_seed(C.SEED)
    device, out = get_device(), ensure_dir(args.out)

    sched = DiffusionSchedule(device=device)
    sanc = SANCNoiseColoring(norm_eps=1e-8).to(device)       # training-time instance
    saliency = load_saliency_model(args.saliency_ckpt, device)
    model = DualBranchReverseUNet().to(device)
    print("Reverse model parameters:", count_parameters(model))

    train_df = get_split(build_combined_df(), "train")
    loader = make_loader(train_df, args.batch_size, shuffle=True)
    print("Train images:", len(train_df), "| batches:", len(loader))

    batch = next(iter(loader))
    with torch.no_grad():
        loss = reverse_training_step(model.eval(), saliency, sanc, sched, batch, device)
    print(f"Sanity check | loss {loss.item():.4f} | finite {torch.isfinite(loss).item()}")
    if args.sanity_only:
        return

    hist = train_reverse_model(model, saliency, sanc, sched, loader, device, args.epochs,
                               C.REVERSE_LR, C.REVERSE_WEIGHT_DECAY, save_path=args.save_path)
    hist.to_csv(out / "reverse_training_history.csv", index=False)


if __name__ == "__main__":
    main()
