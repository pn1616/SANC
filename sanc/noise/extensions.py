"""SANC extensions A, B, C and the 8-arm factorial ablation.

A - Structural fusion      S   = (1 - l_s) M + l_s E,  E = multiscale Sobel edges
B - Uncertainty attenuation S_u = S (1 - l_u U),        U = normalised binary entropy of M
C - Timestep spectral shift w_low += l_t tau, w_high -= l_t tau, tau = t / (T-1)

The reverse model is ALWAYS conditioned on the plain U-Net saliency M; the
extensions only change how the training/sampling noise is coloured.
l_t = 0.05 is the largest value for which every forward-chain increment stays
a valid (non-negative) covariance - see audit A3 (0.10 already fails).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from .sanc import frequency_bands, standardize


# ----------------------------------------------------------------------------
# Component A: multiscale Sobel structure map
# ----------------------------------------------------------------------------
def multiscale_sobel(images, eps=1e-6):
    gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    def edge_strength(x):
        gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + eps)
        mn = mag.amin(dim=(-2, -1), keepdim=True)
        mx = mag.amax(dim=(-2, -1), keepdim=True)
        return (mag - mn) / (mx - mn + eps)

    fine = edge_strength(gray)
    medium = edge_strength(F.avg_pool2d(gray, 3, stride=1, padding=1))
    coarse = edge_strength(F.avg_pool2d(gray, 7, stride=1, padding=3))
    return ((fine + medium + coarse) / 3.0).clamp(0, 1)


# ----------------------------------------------------------------------------
# Component B: entropy uncertainty
# ----------------------------------------------------------------------------
def entropy_uncertainty(M, eps=1e-8):
    """Normalised binary entropy in [0,1]. NaN-safe in float32
    (clamp(eps, 1-eps) is not: 1 - 1e-8 == 1.0 in float32)."""
    U = -(M * torch.log(M + eps) + (1 - M) * torch.log(1 - M + eps)) / torch.log(torch.tensor(2.0, device=M.device))
    return U.clamp(0, 1)


@torch.no_grad()
def compute_extended_saliency(low_light, saliency_model, lambda_s=C.LAMBDA_S, lambda_u=C.LAMBDA_U):
    """Return dict with M, E, S (A), U, S_u (A+B)."""
    M = saliency_model(low_light).clamp(0, 1)
    E = multiscale_sobel(low_light).clamp(0, 1)
    S = ((1 - lambda_s) * M + lambda_s * E).clamp(0, 1)
    U = entropy_uncertainty(M)
    S_u = (S * (1 - lambda_u * U)).clamp(0, 1)
    assert torch.isfinite(U).all() and torch.isfinite(S_u).all(), "non-finite U or S_u"
    return {"M": M, "E": E, "S": S, "U": U, "S_u": S_u}


# ----------------------------------------------------------------------------
# Component C: timestep-dependent SANC
# ----------------------------------------------------------------------------
class ExtendedSANCNoiseColoring(nn.Module):
    def __init__(self, original_sanc, lambda_t=C.LAMBDA_T):
        super().__init__()
        self.original_sanc = original_sanc
        self.lambda_t = lambda_t

    def forward(self, noise, saliency, timestep, T):
        B, _, H, W = noise.shape
        if saliency.shape[-2:] != (H, W):
            saliency = F.interpolate(saliency, size=(H, W), mode="bilinear", align_corners=False)
        M = torch.clamp(saliency, 0.0, 1.0) ** self.original_sanc.saliency_power
        nf = torch.fft.fft2(noise, dim=(-2, -1))
        low_b, mid_b, high_b = frequency_bands(H, W, noise.device)
        low = torch.fft.ifft2(nf * low_b[None, None], dim=(-2, -1)).real
        mid = torch.fft.ifft2(nf * mid_b[None, None], dim=(-2, -1)).real
        high = torch.fft.ifft2(nf * high_b[None, None], dim=(-2, -1)).real

        wl, wm, wh = 0.55 - 0.35 * M, 0.30 * torch.ones_like(M), 0.15 + 0.35 * M
        t = torch.as_tensor(timestep, device=noise.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(B)
        tau = t.reshape(B, 1, 1, 1) / (T - 1)
        wl, wh = wl + self.lambda_t * tau, wh - self.lambda_t * tau
        wl, wm, wh = wl.clamp(min=0), wm.clamp(min=0), wh.clamp(min=0)
        tot = wl + wm + wh + 1e-8
        colored = (wl / tot) * low + (wm / tot) * mid + (wh / tot) * high
        return standardize(colored, 1e-6)


def uncertainty_aware_sanc(noise, saliency, timestep, T, sanc, lambda_u=0.5):
    """First UA-TD-SANC prototype (MEMBER 4 STEP 3): modulates the *saliency*
    with t and U instead of the spectrum. Superseded by ExtendedSANCNoiseColoring;
    kept because its outputs appear in the report."""
    M = torch.clamp(saliency, 0.0, 1.0)
    U = entropy_uncertainty(M)
    t = torch.as_tensor(timestep, device=noise.device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(noise.shape[0])
    t = t.reshape(-1, 1, 1, 1) / (T - 1)
    eff = torch.clamp(M * (1.0 - lambda_u * t * U), 0.0, 1.0)
    return sanc(noise, eff), eff, U


# ----------------------------------------------------------------------------
# 8-arm factorial ablation
# ----------------------------------------------------------------------------
ARMS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def arm_name(a, b, c):
    on = [n for n, f in zip("ABC", (a, b, c)) if f]
    return "SANC (baseline)" if not on else "+" + "+".join(on)


def ablation_noise(z, sal, t, use_A, use_B, use_C, sanc, extended_sanc, T=C.T, lambda_u=C.LAMBDA_U):
    base = sal["S"] if use_A else sal["M"]
    cmap = (base * (1 - lambda_u * sal["U"])).clamp(0, 1) if use_B else base
    if use_C:
        return extended_sanc(z, cmap, t, T)
    return sanc(z, cmap)


class ArmNoise:
    """Noise callable for ``ddim_sample``: ignores the saliency it is passed
    (M, used for conditioning) and returns the colouring of the chosen arm."""

    def __init__(self, sal_dict, t0, arm, sanc, extended_sanc, T=C.T):
        self.sal, self.t0, self.arm = sal_dict, t0, arm
        self.sanc, self.ext, self.T = sanc, extended_sanc, T

    def __call__(self, z, _saliency):
        return ablation_noise(z, self.sal, self.t0, *self.arm, sanc=self.sanc, extended_sanc=self.ext, T=self.T)
