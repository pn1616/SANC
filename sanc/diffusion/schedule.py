"""Linear DDPM schedule (T=100, beta 1e-4 -> 0.02) and SANC forward diffusion."""
import torch

from .. import config as C


def extract(values, timesteps, x_shape):
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class DiffusionSchedule:
    def __init__(self, T=C.T, beta_start=C.BETA_START, beta_end=C.BETA_END, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)

    # Member-3 sampler accesses ``sched.ab``
    @property
    def ab(self):
        return self.alpha_bars

    def q_sample(self, x0, t, colored_noise):
        """x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps_SANC."""
        return (extract(self.sqrt_alpha_bars, t, x0.shape) * x0
                + extract(self.sqrt_one_minus_alpha_bars, t, x0.shape) * colored_noise)
