"""U-Net used for saliency estimation (Experiment A, the selected model).

Parameter names (``enc1.block...``, ``output``) match the Drive checkpoint
``best_unet_saliency.pth``. The notebook had two classes with the same
weights: ``UNet`` (returns logits, used for training) and ``OriginalUNet``
(returns ``sigmoid`` probabilities, used as the saliency map M inside SANC).
Both are covered here through the ``return_probs`` flag.
"""
import torch
import torch.nn as nn

from ..utils import load_checkpoint, state_dict_from


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, return_probs=False):
        super().__init__()
        self.return_probs = return_probs
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.output(d1)
        return torch.sigmoid(logits) if self.return_probs else logits


def load_saliency_model(path, device="cpu", return_probs=True, verbose=True):
    """Load ``best_unet_saliency.pth``. ``return_probs=True`` gives the SANC saliency map M in [0,1]."""
    ckpt = load_checkpoint(path, device)
    model = UNet(3, 1, return_probs=return_probs).to(device)
    model.load_state_dict(state_dict_from(ckpt))
    model.eval()
    if verbose and isinstance(ckpt, dict):
        print(f"Saliency U-Net loaded | epoch={ckpt.get('epoch')} | "
              f"val_dice={ckpt.get('val_dice')} | val_iou={ckpt.get('val_iou')}")
    return model
