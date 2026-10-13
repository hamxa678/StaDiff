"""Staged reverse process (per-pixel DDIM): Phase 1 = ROI only, Phase 2 = whole image."""
import torch

from .diffusion import build_t_map, net_input, q_sample


@torch.no_grad()
def staged_sample(model, x_orig, m, ab, cfg, n_steps=50):
    """
    Corrupt x_orig (ROI -> t_roi, background -> t_bg), then reverse in two phases using the same
    per-pixel timestep map. While s > t_bg the background has t_cur == t_nxt, so it is left untouched
    (Phase 1); once s <= t_bg every pixel is denoised together (Phase 2).
    Returns (reconstruction, corrupted_start).
    """
    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    B, dev = x_orig.shape[0], x_orig.device
    full = lambda v: torch.full((B,), int(v), device=dev, dtype=torch.long)
    grid = torch.linspace(t_roi, 0, n_steps + 1).round().long().tolist()

    x_start, _ = q_sample(x_orig, build_t_map(full(t_roi), m, t_roi, t_bg), ab)
    x = x_start.clone()
    for s_cur, s_nxt in zip(grid[:-1], grid[1:]):
        t_cur = build_t_map(full(s_cur), m, t_roi, t_bg)
        t_nxt = build_t_map(full(s_nxt), m, t_roi, t_bg)
        eps = model(net_input(x, m, t_cur, t_roi), full(s_cur))
        a_c, a_n = ab[t_cur], ab[t_nxt]
        x0_hat = ((x - (1 - a_c).sqrt() * eps) / a_c.sqrt()).clamp(-1, 1)
        eps = (x - a_c.sqrt() * x0_hat) / (1 - a_c).sqrt().clamp(min=1e-8)
        x_new = a_n.sqrt() * x0_hat + (1 - a_n).sqrt() * eps
        x = torch.where(t_nxt == t_cur, x, x_new)        # pixels whose level does not change stay as they are
    return x, x_start
