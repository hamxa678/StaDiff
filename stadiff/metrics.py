"""Image-quality metrics (PSNR per region, SSIM)."""
import torch
import torch.nn.functional as F

to01 = lambda x: ((x + 1) / 2).clamp(0, 1)               # [-1,1] -> [0,1]


def region_psnr(a, b, region):
    """Per-image PSNR over a region. a, b in [0,1]; region (B,1,H,W) in {0,1}. Returns (psnr, valid)."""
    se = ((a - b) ** 2).mean(1, keepdim=True)
    n = region.sum((1, 2, 3))
    mse = (se * region).sum((1, 2, 3)) / n.clamp(min=1)
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10)), n > 0


def ssim(x, y, win=11, sigma=1.5):
    """Per-image SSIM with a Gaussian window. x, y in [0,1], shape (B,C,H,W)."""
    C = x.shape[1]
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(C, 1, win, win).contiguous()
    f = lambda t: F.conv2d(t, w, padding=win // 2, groups=C)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.mean((1, 2, 3))
