"""One-call setup used by the pipeline scripts (replaces the notebook's
'restore everything after the Colab runtime reset' cells)."""
from types import SimpleNamespace

from . import config as C
from .data import build_combined_df, get_split, make_loader
from .diffusion import DiffusionSchedule
from .models import load_reverse_model, load_saliency_model
from .noise import ExtendedSANCNoiseColoring, SANCNoiseColoring
from .utils import get_device


def load_components(saliency_ckpt=None, reverse_ckpt=None, need_reverse=True, need_data=True):
    device = get_device()
    print("Device:", device)
    rt = SimpleNamespace(device=device)
    rt.schedule = DiffusionSchedule(device=device)
    rt.sanc_train = SANCNoiseColoring(norm_eps=1e-8).to(device)   # training-time instance
    rt.sanc_5a = SANCNoiseColoring(norm_eps=1e-6).to(device)      # 'STEP 5A' instance
    rt.extended = ExtendedSANCNoiseColoring(rt.sanc_5a, lambda_t=C.LAMBDA_T).to(device)
    rt.saliency = load_saliency_model(saliency_ckpt or C.CKPT["saliency"], device, return_probs=True)
    rt.reverse = load_reverse_model(reverse_ckpt or C.CKPT["reverse"], device) if need_reverse else None
    if need_data:
        rt.df = build_combined_df()
        rt.train_df, rt.val_df, rt.test_df = (get_split(rt.df, s) for s in ("train", "val", "test"))
        rt.val_loader = make_loader(rt.val_df, shuffle=False)
    return rt
