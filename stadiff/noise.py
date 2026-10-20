"""
Noise generators for the forward process (unit variance, zero mean per image).

  gaussian : white Gaussian noise (standard DDPM). Equal power at every spatial scale, so it destroys fine detail first
             and large structures (e.g. a missing weld) survive until very high noise levels.
  simplex  : multi-octave 2D simplex noise as in AnoDDPM. Spatially correlated, most power at coarse scales,
             so it erases LARGE structures at a much lower noise level. `wavelength` = size (px) of the coarsest octave.
  colored  : Gaussian noise with a 1/f^beta amplitude spectrum. Same coarse-scale effect, exactly Gaussian, GPU-fast.

ROI and background can use different kinds: eps = sqrt(m) * eps_roi + sqrt(1 - m) * eps_bg  (m = soft ROI mask).
E.g. simplex inside the ROI (erase coarse defects) + Gaussian outside (keep the normal layout as clean context).
"""
import math

import numpy as np
import torch

_GRAD = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
_F2, _G2 = 0.5 * (math.sqrt(3.0) - 1.0), (3.0 - math.sqrt(3.0)) / 6.0
_BANK = {}


def _simplex2d(x, y, perm):
    """Vectorised 2D simplex noise (Gustavson). x, y: float arrays of equal shape. Output roughly in [-1, 1]."""
    s = (x + y) * _F2
    i, j = np.floor(x + s).astype(np.int64), np.floor(y + s).astype(np.int64)
    t = (i + j) * _G2
    x0, y0 = x - (i - t), y - (j - t)
    i1 = (x0 > y0).astype(np.int64)
    j1 = 1 - i1
    x1, y1 = x0 - i1 + _G2, y0 - j1 + _G2
    x2, y2 = x0 - 1.0 + 2.0 * _G2, y0 - 1.0 + 2.0 * _G2
    ii, jj = i & 255, j & 255

    def corner(xc, yc, di, dj):
        g = _GRAD[perm[(ii + di + perm[(jj + dj) & 255]) & 255] & 7]
        tt = np.maximum(0.5 - xc * xc - yc * yc, 0.0)
        return (tt ** 4) * (g[..., 0] * xc + g[..., 1] * yc)

    return 70.0 * (corner(x0, y0, 0, 0) + corner(x1, y1, i1, j1) + corner(x2, y2, 1, 1))


def _simplex_field(H, W, rng, wavelength, octaves, persistence):
    perm = rng.permutation(256)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ox, oy = rng.uniform(0, 256, 2)
    out, amp, freq = np.zeros((H, W)), 1.0, 1.0 / wavelength
    for _ in range(octaves):
        out += amp * _simplex2d(xx * freq + ox, yy * freq + oy, perm)
        amp, freq = amp * persistence, freq * 2.0
    return out


def _simplex_bank(H, W, nz, device, n=64):
    """A bank of n large simplex fields (1.5x image size). Batches are random crops / flips / sign flips of it."""
    key = (H, W, float(nz["wavelength"]), int(nz["octaves"]), float(nz["persistence"]), str(device))
    if key not in _BANK:
        rng = np.random.RandomState(1234)
        Hb, Wb = int(H * 1.5), int(W * 1.5)
        fields = np.stack([_simplex_field(Hb, Wb, rng, nz["wavelength"], nz["octaves"], nz["persistence"]) for _ in range(n)])
        _BANK[key] = torch.from_numpy(fields).half().to(device)
    return _BANK[key]


def _simplex(like, nz):
    B, C, H, W = like.shape
    bank = _simplex_bank(H, W, nz, like.device)
    n, Hb, Wb = bank.shape
    out = torch.empty_like(like)
    for b in range(B):
        for c in range(C):
            k, y, x = int(torch.randint(n, (1,))), int(torch.randint(Hb - H + 1, (1,))), int(torch.randint(Wb - W + 1, (1,)))
            f = bank[k, y:y + H, x:x + W].to(like.dtype)
            if torch.rand(1) < 0.5:
                f = f.flip(0)
            if torch.rand(1) < 0.5:
                f = f.flip(1)
            if torch.rand(1) < 0.5:
                f = -f
            out[b, c] = (f - f.mean()) / (f.std() + 1e-6)
    return out


def _colored(like, beta):
    H, W = like.shape[-2:]
    fy = torch.fft.fftfreq(H, device=like.device)[:, None]
    fx = torch.fft.fftfreq(W, device=like.device)[None, :]
    r = torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1.0 / max(H, W))
    amp = r ** (-beta)
    amp[0, 0] = 0.0                                              # zero mean
    out = torch.fft.ifft2(torch.fft.fft2(torch.randn_like(like)) * amp).real
    mean, std = out.mean((-2, -1), keepdim=True), out.std((-2, -1), keepdim=True)
    return (out - mean) / (std + 1e-6)


def _make(kind, like, nz):
    if kind == "gaussian":
        return torch.randn_like(like)
    if kind == "simplex":
        return _simplex(like, nz)
    if kind == "colored":
        return _colored(like, float(nz["beta"]))
    raise ValueError(f"unknown noise kind {kind!r}")


def sample_noise(like, m, cfg):
    """Noise for the forward process. like: (B,C,H,W) image batch, m: (B,1,H,W) soft ROI mask, cfg['noise'] settings."""
    nz = cfg.get("noise") or {"roi": "gaussian", "bg": "gaussian"}
    kr, kb = nz["roi"], nz["bg"]
    if kr == "gaussian" and kb == "gaussian":
        return torch.randn_like(like)
    e_roi = _make(kr, like, nz)
    if kr == kb:
        return e_roi
    return m.sqrt() * e_roi + (1 - m).sqrt() * _make(kb, like, nz)
