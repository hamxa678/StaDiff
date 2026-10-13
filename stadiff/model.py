"""Mask- and timestep-map-conditioned U-Net, and an EMA wrapper."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the scalar global step s."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.t(temb)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        return self.c2(F.silu(h)) + self.skip(x)


class Attn(nn.Module):
    """Self-attention at the bottleneck: lets the ROI look at the whole (cleaner) background."""

    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).transpose(2, 3).unbind(1)
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        return x + self.o(h.transpose(1, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    """in: [x_t(3), mask(1), t_map(1)] -> predicted noise (3). Image size must be divisible by 4."""

    def __init__(self, in_ch=5, out_ch=3, base=64, mult=(1, 2, 2), tdim=256):
        super().__init__()
        self.base = base
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        chs = [base * m for m in mult]
        self.inc = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down, self.downs = nn.ModuleList(), nn.ModuleList()
        c = chs[0]
        for i, co in enumerate(chs):
            self.down.append(ResBlock(c, co, tdim))
            c = co
            self.downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
        self.mid1, self.attn, self.mid2 = ResBlock(c, c, tdim), Attn(c), ResBlock(c, c, tdim)
        self.up, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(chs))):
            self.up.append(ResBlock(c + co, co, tdim))
            c = co
            self.ups.append(nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c, c, 3, padding=1))
                            if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))

    def forward(self, x, s):
        temb = self.temb(timestep_embedding(s, self.base))
        h, skips = self.inc(x), []
        for blk, ds in zip(self.down, self.downs):
            h = blk(h, temb)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)
        for blk, us in zip(self.up, self.ups):
            h = us(blk(torch.cat([h, skips.pop()], 1), temb))
        return self.out(h)


class EMA:
    """Exponential moving average of the weights (used for validation, testing and inference)."""

    def __init__(self, model, decay):
        self.m, self.d = copy.deepcopy(model).eval().requires_grad_(False), decay

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.d, (1 + step) / (10 + step))        # warm-up so short runs still get a useful EMA
        for pe, p in zip(self.m.parameters(), model.parameters()):
            pe.lerp_(p, 1 - d)
