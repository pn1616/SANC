"""Segmentation and image-quality metrics."""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- segmentation
def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Batch-mean Dice from logits (used for validation during training)."""
    preds = (torch.sigmoid(logits) >= threshold).float().view(logits.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (preds * t).sum(1)
    return ((2 * inter + smooth) / (preds.sum(1) + t.sum(1) + smooth)).mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) >= threshold).float().view(logits.size(0), -1)
    t = targets.view(targets.size(0), -1)
    inter = (preds * t).sum(1)
    union = preds.sum(1) + t.sum(1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def per_image_dice_iou(probs, targets, threshold=0.5, eps=1e-7):
    """Per-image Dice and IoU tensors (used for the final per-dataset test table)."""
    pred = (probs >= threshold).float()
    y = (targets >= 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * y).sum(dims)
    ps, ys = pred.sum(dims), y.sum(dims)
    dice = (2 * inter + eps) / (ps + ys + eps)
    iou = (inter + eps) / (ps + ys - inter + eps)
    return dice, iou


def soft_dice(probs, targets, smooth=1e-6):
    inter = (probs * targets).sum((1, 2, 3))
    return (2 * inter + smooth) / (probs.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + smooth)


# ---------------------------------------------------------------- image quality
def psnr_batch(out, ref):
    mse = (out.clamp(0, 1) - ref).pow(2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse)


_SSIM_CACHE = {}


def _ssim_window(device, channels=3, size=11, sigma=1.5):
    key = (str(device), channels)
    if key not in _SSIM_CACHE:
        g = torch.exp(-((torch.arange(size, device=device) - size // 2).float() ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _SSIM_CACHE[key] = (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()
    return _SSIM_CACHE[key]


def ssim_batch(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """Gaussian 11x11 (sigma 1.5) SSIM, valid region - same as ablation STEP 15."""
    x = x.clamp(0, 1)
    W = _ssim_window(x.device, x.shape[1])
    f = lambda z: F.conv2d(z, W, groups=x.shape[1])  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return m.mean(dim=(1, 2, 3))
