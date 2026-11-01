"""Pipeline 03 - Saliency U-Net training (Experiments A-F).

| Exp | Model          | Loss                    | Extras                          | Best val Dice |
|-----|----------------|-------------------------|---------------------------------|---------------|
| A   | U-Net          | BCE + global Dice       | 20 ep, Adam 1e-4  (SELECTED)    | 0.7545        |
| B   | U-Net          | 0.5 BCE + 0.5 Dice      | 20 ep                           | 0.7507        |
| C   | U-Net          | Tversky (a .4, b .6)    | augmentation, 30 ep, patience 7 | 0.7502        |
| D   | Attention U-Net| BCE                     | 20 ep                           | 0.7199        |
| E   | UNet++         | BCE                     | 20 ep                           | 0.7315        |
| F   | U-Net (from A) | BCE                     | fine-tune lr 1e-5, 15 ep, pat 5 | no gain       |

Usage:
    python pipelines/03_saliency_unet/train.py --experiment A
    python pipelines/03_saliency_unet/train.py --experiment F   # needs A's checkpoint
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sanc import config as C  # noqa: E402
from sanc.data import build_combined_df, get_split, make_loader  # noqa: E402
from sanc.losses import BCEDiceLoss, TverskyLoss, WeightedBCEDiceLoss  # noqa: E402
from sanc.models import build_segmentation_model  # noqa: E402
from sanc.training import train_segmentation  # noqa: E402
from sanc.utils import ensure_dir, get_device, load_checkpoint, set_seed, state_dict_from  # noqa: E402

EXPERIMENTS = {
    "A": dict(arch="unet", loss=BCEDiceLoss, epochs=20, lr=1e-4, patience=None, aug=False, ckpt="saliency"),
    "B": dict(arch="unet", loss=WeightedBCEDiceLoss, epochs=20, lr=1e-4, patience=None, aug=False, ckpt="saliency_bce_dice"),
    "C": dict(arch="unet", loss=TverskyLoss, epochs=30, lr=1e-4, patience=7, aug=True, ckpt="saliency_tversky_aug"),
    "D": dict(arch="attention_unet", loss=nn.BCEWithLogitsLoss, epochs=20, lr=1e-4, patience=None, aug=False, ckpt="attention_unet"),
    "E": dict(arch="unetpp", loss=nn.BCEWithLogitsLoss, epochs=20, lr=1e-4, patience=None, aug=False, ckpt="unetpp"),
    "F": dict(arch="unet", loss=nn.BCEWithLogitsLoss, epochs=15, lr=1e-5, patience=5, aug=False, ckpt="saliency_finetuned",
              init_from="saliency"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="A")
    ap.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--save-path", default=None, help="override checkpoint path")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "03_saliency_unet"))
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    set_seed(C.SEED)
    device = get_device()
    out = ensure_dir(args.out)
    save_path = Path(args.save_path) if args.save_path else C.CKPT[cfg["ckpt"]]

    df = build_combined_df()
    train_df, val_df = get_split(df, "train"), get_split(df, "val")
    train_loader = make_loader(train_df, args.batch_size, shuffle=True, augmented=cfg["aug"], training=True)
    val_loader = make_loader(val_df, args.batch_size, shuffle=False)
    print(f"Experiment {args.experiment} | train {len(train_df)} | val {len(val_df)} | device {device}")

    model = build_segmentation_model(cfg["arch"]).to(device)
    best, offset = -1.0, 0
    if "init_from" in cfg:
        ckpt = load_checkpoint(C.CKPT[cfg["init_from"]], device)
        model.load_state_dict(state_dict_from(ckpt))
        best, offset = float(ckpt.get("val_dice", 0.0)), int(ckpt.get("epoch", 0))
        print(f"Fine-tuning from epoch {offset}, starting val Dice {best:.4f} (saved only if improved)")

    criterion = cfg["loss"]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history, best = train_segmentation(model, criterion, optimizer, train_loader, val_loader, device,
                                       args.epochs or cfg["epochs"], save_path, patience=cfg["patience"],
                                       best_val_dice=best, epoch_offset=offset)
    history.to_csv(out / f"history_experiment_{args.experiment}.csv", index=False)
    print(f"\nBest validation Dice: {best:.4f} | checkpoint: {save_path}")


if __name__ == "__main__":
    main()
