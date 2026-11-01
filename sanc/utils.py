"""Small shared helpers."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def load_checkpoint(path, device="cpu"):
    """torch.load that works on old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


def state_dict_from(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    return checkpoint


def tensor_to_image(t):
    """[1,C,H,W] or [C,H,W] tensor -> HxWxC numpy in [0,1]."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def describe(name, x):
    """One-line tensor statistics used throughout the audits."""
    return (f"{name:24s} | shape={tuple(x.shape)} | mean={x.mean().item():.5f} | "
            f"std={x.std().item():.5f} | min={x.min().item():.4f} | max={x.max().item():.4f} | "
            f"finite={torch.isfinite(x).all().item()}")


def is_valid_file(p):
    return isinstance(p, str) and os.path.isfile(p) and os.path.getsize(p) > 0
