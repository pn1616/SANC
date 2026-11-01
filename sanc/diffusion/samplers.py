"""Reverse samplers.

* ``ddpm_sample``  - ancestral DDPM from pure noise (variance='beta' is the first
  sampler, 'posterior' the corrected v2). Approximation for SANC because the
  reverse step re-injects white noise.
* ``ddim_sample``  - deterministic (eta=0) DDIM started at t0 from the
  low-light image, with x0 clipping. Copied verbatim from
  SANC_Member3_Evaluation.ipynb; this is the sampler frozen for the ablation
  (t0 = 99, 25 steps).
"""
import torch

from ..noise.baselines import blue_colour


@torch.no_grad()
def ddpm_sample(low_light, reverse_model, saliency_model, schedule, variance="posterior",
                start_from_input=False, return_trajectory=None):
    reverse_model.eval()
    saliency_model.eval()
    B = low_light.shape[0]
    M = saliency_model(low_light)
    x = low_light.clone() if start_from_input else torch.randn_like(low_light)
    traj = {}
    for step in reversed(range(schedule.T)):
        t = torch.full((B,), step, device=low_light.device, dtype=torch.long)
        eps = reverse_model(x, low_light, M, t)
        beta, alpha, ab = schedule.betas[step], schedule.alphas[step], schedule.alpha_bars[step]
        mean = (x - (beta / torch.sqrt(1.0 - ab)) * eps) / torch.sqrt(alpha)
        if step > 0:
            var = schedule.posterior_variance[step] if variance == "posterior" else beta
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean
        if return_trajectory and step in return_trajectory:
            traj[step] = x.detach().clone()
    return (x, traj) if return_trajectory else x


def make_noise(kind, shape, saliency, gen=None, sanc=None):
    dev = saliency.device
    z = torch.randn(shape, generator=gen, device=dev) if gen is not None else torch.randn(shape, device=dev)
    if kind == "gaussian":
        return z
    if kind == "blue":
        return blue_colour(z)
    if kind == "sanc":
        return sanc(z, saliency)
    raise ValueError(kind)


@torch.no_grad()
def ddim_sample(model, low, sal, kind, sanc, sched, t0, n_steps, gen=None, clip_x0=True, init="lowlight"):
    """Deterministic (eta=0) sampler, started at step t0 from the low-light image."""
    model.eval()
    B = low.shape[0]
    ts = torch.linspace(t0, 0, max(2, n_steps)).round().long().tolist()
    ts = sorted(set(ts), reverse=True)
    eps0 = make_noise(kind, low.shape, sal, gen, sanc)
    x0_init = low if init == "lowlight" else torch.zeros_like(low)
    ab0 = sched.ab[t0]
    x = ab0.sqrt() * x0_init + (1.0 - ab0).sqrt() * eps0
    for i, t in enumerate(ts):
        tt = torch.full((B,), t, device=low.device, dtype=torch.long)
        eps = model(x, low, sal, tt)
        ab = sched.ab[t]
        x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
        if clip_x0:
            x0 = x0.clamp(0.0, 1.0)
            eps = (x - ab.sqrt() * x0) / (1.0 - ab).sqrt()
        if i + 1 < len(ts):
            abp = sched.ab[ts[i + 1]]
            x = abp.sqrt() * x0 + (1.0 - abp).sqrt() * eps
        else:
            x = x0
    return x
