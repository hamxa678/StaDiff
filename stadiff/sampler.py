"""Staged reverse process (per-pixel DDIM): Phase 1 = ROI only, Phase 2 = whole image."""
import torch

from .diffusion import build_t_map, net_input, q_sample
from .noise import sample_noise


@torch.no_grad()
def staged_sample(model, x_orig, m, ab, cfg, n_steps=50, return_traj=False, s_start=None):
    """
    Corrupt x_orig (ROI -> t_roi, background -> t_bg), then reverse in two phases using the same
    per-pixel timestep map. While s > t_bg the background has t_cur == t_nxt, so it is left untouched
    (Phase 1); once s <= t_bg every pixel is denoised together (Phase 2).
    s_start: global step to start from (default t_roi). s_start=t_bg gives a uniform start (Phase 2 only), used by diagnose_gap.py.
    Returns (reconstruction, corrupted_start)  [+ trajectory list of (s, x) if return_traj=True].
    """
    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    B, dev = x_orig.shape[0], x_orig.device
    full = lambda v: torch.full((B,), int(v), device=dev, dtype=torch.long)
    s0 = t_roi if s_start is None else int(s_start)
    grid = torch.linspace(s0, 0, n_steps + 1).round().long().tolist()

    x_start, _ = q_sample(x_orig, build_t_map(full(s0), m, t_roi, t_bg), ab, sample_noise(x_orig, m, cfg))
    x = x_start.clone()
    traj = [(grid[0], x.clone())] if return_traj else None
    for s_cur, s_nxt in zip(grid[:-1], grid[1:]):
        t_cur = build_t_map(full(s_cur), m, t_roi, t_bg)
        t_nxt = build_t_map(full(s_nxt), m, t_roi, t_bg)
        eps = model(net_input(x, m, t_cur, t_roi), full(s_cur))      # the model READS the whole image...
        a_c, a_n = ab[t_cur], ab[t_nxt]
        x0_hat = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)
        eps = (x - a_c.sqrt() * x0_hat) / (1 - a_c).sqrt().clamp(min=1e-8)
        x_new = a_n.sqrt() * x0_hat + (1 - a_n).sqrt() * eps
        x = torch.where(t_nxt == t_cur, x, x_new)                    # ...but only pixels whose level changes are updated
        if return_traj:
            traj.append((s_nxt, x.clone()))
    return (x, x_start, traj) if return_traj else (x, x_start)
