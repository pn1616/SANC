"""Baseline noise types for the sampler (``make_noise``).

NOTE (generated): ``blue_colour`` is referenced by the Member-3 sampler copied
into the notebook (``make_noise(kind="blue")``) but its definition lives in
``SANC_Member3_Evaluation.ipynb``, which was not part of this notebook. This is
a standard stand-in: amplitude proportional to radial frequency (power ~ f^2),
re-standardised to N(0,1). Replace it with Member 3's version if it differs.
"""
import torch


def blue_colour(z):
    H, W = z.shape[-2:]
    fy = torch.fft.fftfreq(H, device=z.device)
    fx = torch.fft.fftfreq(W, device=z.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    r = r / (r.max() + 1e-8)
    out = torch.fft.ifft2(torch.fft.fft2(z, dim=(-2, -1)) * r[None, None], dim=(-2, -1)).real
    mean = out.mean(dim=(-2, -1), keepdim=True)
    std = out.std(dim=(-2, -1), keepdim=True)
    return (out - mean) / (std + 1e-8)
