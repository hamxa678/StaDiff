"""ROI masks: random (training / evaluation) and user-given boxes (inference)."""
import math

import torch
import torch.nn.functional as F


def gaussian_blur(m, sigma):
    """Separable Gaussian blur of a (B,1,H,W) mask -> soft border."""
    if sigma <= 0:
        return m
    k = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(k, device=m.device, dtype=m.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    m = F.conv2d(F.pad(m, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(m, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))


def random_masks(B, H, W, device, gen=None):
    """Random soft rectangles / ellipses anywhere in the image. Pass `gen` for reproducible masks."""
    r = lambda *shape: torch.rand(*shape, generator=gen).to(device)
    ys = torch.linspace(0, 1, H, device=device).view(1, H, 1)
    xs = torch.linspace(0, 1, W, device=device).view(1, 1, W)
    cx, cy = r(B, 1, 1), r(B, 1, 1)
    w, h = 0.2 + 0.3 * r(B, 1, 1), 0.2 + 0.3 * r(B, 1, 1)
    rect = ((xs - cx).abs() < w / 2) & ((ys - cy).abs() < h / 2)
    ell = (((xs - cx) / (w / 2)) ** 2 + ((ys - cy) / (h / 2)) ** 2) < 1
    m = torch.where(r(B, 1, 1) < 0.5, ell, rect).float().unsqueeze(1)
    sigma = (H / 32) * (0.5 + float(torch.rand(1, generator=gen)))
    return gaussian_blur(m, sigma)


def box_mask(box, H, W, device):
    """Rectangle from fractional coords (x0, y0, x1, y1) -> soft mask (1,1,H,W)."""
    x0, y0, x1, y1 = box
    m = torch.zeros(1, 1, H, W, device=device)
    m[..., int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 1.0
    return gaussian_blur(m, H / 32)
