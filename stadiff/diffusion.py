"""Noise schedule, per-pixel timestep map and the forward (noising) process."""
import torch

from .config import T_SCHED


def make_ab(device):
    """alpha_bar[t] for t = 0..T_SCHED   (alpha_bar[0] = 1). Standard linear DDPM schedule."""
    betas = torch.linspace(1e-4, 0.02, T_SCHED, dtype=torch.float64)
    ab = torch.cumprod(1.0 - betas, 0)
    return torch.cat([torch.ones(1, dtype=torch.float64), ab]).float().to(device)


def build_t_map(s, m, t_roi, t_bg):
    """
    Per-pixel timestep map.
      s : (B,) global step in [0, t_roi]
      m : (B,1,H,W) soft ROI mask in [0, 1]
    ROI pixels follow s up to t_roi; background pixels stop at t_bg:
      t_i = cap + m_i * (s - cap),   cap = min(s, t_bg)
    """
    s4 = s.view(-1, 1, 1, 1).float()
    cap = s4.clamp(max=t_bg)
    return (cap + m * (s4 - cap)).round().long()


def q_sample(x0, t_map, ab, noise=None):
    """Per-pixel forward process: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * eps."""
    if noise is None:
        noise = torch.randn_like(x0)
    a = ab[t_map]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise


def net_input(x_t, m, t_map, t_roi):
    """What the network sees: noisy image, ROI mask, normalised per-pixel timestep (5 channels)."""
    return torch.cat([x_t, m, t_map.float() / t_roi], dim=1)
