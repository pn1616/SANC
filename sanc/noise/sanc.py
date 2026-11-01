"""SANC: Saliency-Adaptive Noise Coloring (FFT formulation).

Gaussian noise z is split in the Fourier domain into three smooth radial
bands (low / mid / high, Gaussian bumps at r = 0.15, 0.45, 0.80 with
bandwidth 0.20, normalised to a partition of unity). The three spatial
components are blended *per pixel* with saliency-dependent weights

    w_low  = 0.55 - 0.35 M
    w_mid  = 0.30
    w_high = 0.15 + 0.35 M          (weights always sum to 1)

so salient (structural) regions receive high-frequency noise and background
regions low-frequency noise. The result is re-standardised to zero mean and
unit variance per sample/channel so it is drop-in compatible with the DDPM
schedule.

``norm_eps``: the training-time instance (``sanc_noise_coloring``, cell 234)
used 1e-8; the earlier "STEP 5A" instance (``sanc``) used 1e-6. Audit A1 shows
the two agree to 1.6e-5, so both are exposed via this one class.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_CENTERS = (0.15, 0.45, 0.80)
BANDWIDTH = 0.20


def frequency_bands(H, W, device, centers=BAND_CENTERS, bandwidth=BANDWIDTH):
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    r = r / (r.max() + 1e-8)
    bands = [torch.exp(-0.5 * ((r - c) / bandwidth) ** 2) for c in centers]
    total = sum(bands) + 1e-8
    return tuple(b / total for b in bands)


def split_bands(noise):
    """Return (low, mid, high) spatial noise components of ``noise``."""
    H, W = noise.shape[-2:]
    low_b, mid_b, high_b = frequency_bands(H, W, noise.device)
    nf = torch.fft.fft2(noise, dim=(-2, -1))
    return tuple(torch.fft.ifft2(nf * b[None, None], dim=(-2, -1)).real for b in (low_b, mid_b, high_b))


def sanc_weights(M):
    return 0.55 - 0.35 * M, 0.30 * torch.ones_like(M), 0.15 + 0.35 * M


def standardize(x, eps):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    return (x - mean) / (std + eps)


class SANCNoiseColoring(nn.Module):
    def __init__(self, saliency_power=1.0, norm_eps=1e-8):
        super().__init__()
        self.saliency_power = saliency_power
        self.norm_eps = norm_eps

    def _frequency_bands(self, H, W, device):   # kept for API compatibility with the notebook
        return frequency_bands(H, W, device)

    def prepare_saliency(self, saliency, H, W):
        if saliency.shape[-2:] != (H, W):
            saliency = F.interpolate(saliency, size=(H, W), mode="bilinear", align_corners=False)
        return torch.clamp(saliency, 0.0, 1.0) ** self.saliency_power

    def forward(self, noise, saliency):
        H, W = noise.shape[-2:]
        M = self.prepare_saliency(saliency, H, W)
        low, mid, high = split_bands(noise)
        wl, wm, wh = sanc_weights(M)
        return standardize(wl * low + wm * mid + wh * high, self.norm_eps)


# ----------------------------------------------------------------------------
# Spectral diagnostics used in the SANC tests / audits
# ----------------------------------------------------------------------------
def radial_energy(image_2d):
    """Mean |FFT| in inner third vs outer third of the (shifted) spectrum."""
    img = image_2d.detach().cpu().numpy()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    H, W = img.shape
    y, x = np.indices((H, W))
    r = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
    rmax = min(H, W) // 2
    return mag[r < rmax * 0.33].mean(), mag[r > rmax * 0.66].mean()


def high_frequency_ratio(x, cutoff=0.60):
    H, W = x.shape[-2:]
    power = torch.fft.fft2(x, dim=(-2, -1)).abs().square()
    fy, fx = torch.fft.fftfreq(H, device=x.device), torch.fft.fftfreq(W, device=x.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    hf = (power * (r > cutoff * r.max())[None, None]).sum((-2, -1))
    return (hf / (power.sum((-2, -1)) + 1e-8)).mean().item()


def band_energy_shares(x):
    """Fraction of spectral power falling in the SANC low / mid / high bands."""
    low_b, mid_b, high_b = frequency_bands(*x.shape[-2:], x.device)
    p = torch.fft.fft2(x, dim=(-2, -1)).abs().square()
    e = [(p * b[None, None]).sum() for b in (low_b, mid_b, high_b)]
    tot = sum(e)
    return [(v / tot).item() for v in e]


def local_high_frequency_energy(noise_2d, region):
    """Mean squared finite difference of the noise inside a boolean region."""
    dx = noise_2d[:, 1:] - noise_2d[:, :-1]
    dy = noise_2d[1:, :] - noise_2d[:-1, :]
    rx = region[:, 1:] & region[:, :-1]
    ry = region[1:, :] & region[:-1, :]
    parts = []
    if rx.any():
        parts.append((dx[rx] ** 2).mean())
    if ry.any():
        parts.append((dy[ry] ** 2).mean())
    return float("nan") if not parts else torch.stack(parts).mean().item()
