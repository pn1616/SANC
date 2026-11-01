"""Segmentation losses used in the saliency experiments."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Global (whole-batch) soft Dice - used in Experiment A."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    """Experiment A (selected model): BCE + global Dice, unweighted."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class WeightedBCEDiceLoss(nn.Module):
    """Experiment B: 0.5 * BCE + 0.5 * per-image Dice."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits).view(logits.size(0), -1)
        t = targets.view(targets.size(0), -1)
        dice = (2 * (p * t).sum(1) + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)
        return self.bce_weight * bce + self.dice_weight * (1 - dice.mean())


class TverskyLoss(nn.Module):
    """Experiment C: Tversky (alpha=0.4 FP, beta=0.6 FN)."""

    def __init__(self, alpha=0.4, beta=0.6, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.size(0), -1)
        t = targets.view(targets.size(0), -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tv.mean()
