from .unet import UNet, load_saliency_model  # noqa: F401
from .attention_unet import AttentionUNet  # noqa: F401
from .unetpp import UNetPlusPlus  # noqa: F401
from .reverse_unet import DualBranchReverseUNet, load_reverse_model  # noqa: F401


def build_segmentation_model(arch="unet"):
    return {"unet": UNet, "attention_unet": AttentionUNet, "unetpp": UNetPlusPlus}[arch](in_channels=3, out_channels=1)
