"""Training / evaluation loops shared by the saliency, reverse-diffusion and LODO pipelines."""
import os

import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import dice_score, iou_score, per_image_dice_iou


# ============================================================================
# Saliency segmentation
# ============================================================================
def train_segmentation(model, criterion, optimizer, train_loader, val_loader, device, epochs,
                       save_path, patience=None, best_val_dice=-1.0, epoch_offset=0, log=print):
    """Train with best-val-Dice checkpointing (+ optional early stopping).

    Checkpoint format is identical to the notebook:
    ``{epoch, model_state_dict, optimizer_state_dict, val_loss, val_dice, val_iou}``.
    """
    os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
    history, stale = [], 0
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for b in train_loader:
            x, y = b["low_light"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
        tl /= max(1, len(train_loader))

        vl, vd, vi = evaluate_segmentation(model, criterion, val_loader, device)
        best = vd > best_val_dice
        if best:
            best_val_dice, stale = vd, 0
            torch.save({"epoch": epoch_offset + epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": vl, "val_dice": vd, "val_iou": vi}, save_path)
        else:
            stale += 1
        history.append(dict(epoch=epoch_offset + epoch + 1, train_loss=tl, val_loss=vl, val_dice=vd, val_iou=vi, best=best))
        log(f"Epoch [{epoch + 1:02d}/{epochs}] Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | "
            f"Val Dice: {vd:.4f} | Val IoU: {vi:.4f}" + (" | BEST" if best else ""))
        if patience is not None and stale >= patience:
            log("Early stopping triggered.")
            break
    return pd.DataFrame(history), best_val_dice


@torch.no_grad()
def evaluate_segmentation(model, criterion, loader, device):
    model.eval()
    tl = td = ti = 0.0
    for b in loader:
        x, y = b["low_light"].to(device), b["mask"].to(device)
        logits = model(x)
        tl += criterion(logits, y).item()
        td += dice_score(logits, y)
        ti += iou_score(logits, y)
    n = max(1, len(loader))
    return tl / n, td / n, ti / n


@torch.no_grad()
def per_dataset_test_metrics(model, loader, device, logits=True, threshold=0.5):
    """Per-image Dice/IoU aggregated per dataset (the reported saliency table)."""
    model.eval()
    rows = []
    for b in loader:
        x, y = b["low_light"].to(device), b["mask"].to(device)
        out = model(x)
        probs = torch.sigmoid(out) if logits else out
        d, j = per_image_dice_iou(probs, y, threshold)
        for name, fn, dd, jj in zip(b["dataset"], b["filename"], d.cpu().tolist(), j.cpu().tolist()):
            rows.append(dict(dataset=name, filename=fn, dice=dd, iou=jj))
    per_image = pd.DataFrame(rows)
    summary = (per_image.groupby("dataset").agg(Images_Evaluated=("dice", "size"),
                                                Mean_Dice=("dice", "mean"), Mean_IoU=("iou", "mean"))
               .reset_index().rename(columns={"dataset": "Dataset"}))
    return per_image, summary


# ============================================================================
# Reverse diffusion
# ============================================================================
def reverse_training_step(reverse_model, saliency_model, sanc, schedule, batch, device, noise_fn=None):
    """One SANC noise-prediction step: returns the MSE loss tensor.

    1. M = saliency(low-light)  2. z ~ N(0, I)  3. eps = SANC(z, M)
    4. t ~ U{0..T-1}  5. x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps  6. loss = ||eps_theta - eps||^2
    """
    low, x0 = batch["low_light"].to(device), batch["original"].to(device)
    with torch.no_grad():
        M = saliency_model(low)
        z = torch.randn_like(x0)
        eps = noise_fn(z, M) if noise_fn is not None else sanc(z, M)
    t = torch.randint(0, schedule.T, (x0.size(0),), device=device).long()
    x_t = schedule.q_sample(x0, t, eps)
    pred = reverse_model(x_t, low, M, t)
    return F.mse_loss(pred, eps)


def train_reverse_model(reverse_model, saliency_model, sanc, schedule, train_loader, device, epochs,
                        lr, weight_decay, save_path=None, log=print, noise_fn=None):
    optimizer = torch.optim.AdamW(reverse_model.parameters(), lr=lr, weight_decay=weight_decay)
    saliency_model.eval()
    history = []
    for epoch in range(epochs):
        reverse_model.train()
        running = 0.0
        for b in train_loader:
            loss = reverse_training_step(reverse_model, saliency_model, sanc, schedule, b, device, noise_fn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        ep_loss = running / max(1, len(train_loader))
        history.append(dict(epoch=epoch + 1, train_loss=ep_loss))
        log(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {ep_loss:.6f}")
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
        torch.save({"model_state_dict": reverse_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_epochs": epochs, "loss": history[-1]["train_loss"] if history else None}, save_path)
        log(f"Saved: {save_path}")
    return pd.DataFrame(history)
