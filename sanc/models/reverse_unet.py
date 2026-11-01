"""Conditional dual-branch reverse-diffusion U-Net.

Input  : x_t (3) + low-light (3) + saliency M (1) = 7 channels, timestep t
Output : predicted SANC-coloured noise (3 channels)

Shared encoder -> illumination decoder + detail decoder -> attention fusion.
Parameter names match ``reverse_diffusion_model.pth`` (strict loading passes).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from ..utils import load_checkpoint, state_dict_from


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2), nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, t):
        half = self.embedding_dim // 2
        exponent = -np.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        emb = t.float().unsqueeze(1) * torch.exp(exponent)
        return self.mlp(torch.cat([torch.sin(emb), torch.cos(emb)], dim=1))


class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.activation = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        h = self.activation(h + self.time_proj(t_emb)[:, :, None, None])
        h = self.norm2(self.conv2(h))
        return self.activation(h + self.skip(x))


class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), nn.SiLU(),
            nn.Conv2d(channels, 2, 1), nn.Softmax(dim=1),
        )
        self.output = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, illumination, detail):
        w = self.attention(torch.cat([illumination, detail], dim=1))
        return self.output(w[:, 0:1] * illumination + w[:, 1:2] * detail)


class DualBranchReverseUNet(nn.Module):
    def __init__(self, in_channels=C.REVERSE_IN_CHANNELS, base_channels=C.REVERSE_BASE_CHANNELS,
                 time_dim=C.REVERSE_TIME_DIM):
        super().__init__()
        b = base_channels
        self.time_embedding = TimeEmbedding(time_dim)
        self.enc1 = TimeResBlock(in_channels, b, time_dim)
        self.enc2 = TimeResBlock(b, b * 2, time_dim)
        self.enc3 = TimeResBlock(b * 2, b * 4, time_dim)
        self.bottleneck = TimeResBlock(b * 4, b * 8, time_dim)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.illum_dec3 = TimeResBlock(b * 8, b * 4, time_dim)
        self.illum_dec2 = TimeResBlock(b * 4, b * 2, time_dim)
        self.illum_dec1 = TimeResBlock(b * 2, b, time_dim)
        self.detail_dec3 = TimeResBlock(b * 8, b * 4, time_dim)
        self.detail_dec2 = TimeResBlock(b * 4, b * 2, time_dim)
        self.detail_dec1 = TimeResBlock(b * 2, b, time_dim)
        self.fusion = AttentionFusion(b)
        self.output = nn.Conv2d(b, 3, 1)

    def _decode(self, b, e1, e2, e3, t, d3, d2, d1):
        x = d3(torch.cat([self.up3(b), e3], 1), t)
        x = d2(torch.cat([self.up2(x), e2], 1), t)
        return d1(torch.cat([self.up1(x), e1], 1), t)

    def forward(self, x_t, low_light, saliency, t):
        x = torch.cat([x_t, low_light, saliency], dim=1)
        t_emb = self.time_embedding(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(F.avg_pool2d(e1, 2), t_emb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), t_emb)
        b = self.bottleneck(F.avg_pool2d(e3, 2), t_emb)
        illum = self._decode(b, e1, e2, e3, t_emb, self.illum_dec3, self.illum_dec2, self.illum_dec1)
        detail = self._decode(b, e1, e2, e3, t_emb, self.detail_dec3, self.detail_dec2, self.detail_dec1)
        return self.output(self.fusion(illum, detail))


def load_reverse_model(path, device="cpu", strict=True, verbose=True):
    ckpt = load_checkpoint(path, device)
    model = DualBranchReverseUNet().to(device)
    result = model.load_state_dict(state_dict_from(ckpt), strict=strict)
    model.eval()
    if verbose:
        print(f"Reverse model loaded | strict={strict} | {result} | epochs={ckpt.get('num_epochs')}")
    return model
