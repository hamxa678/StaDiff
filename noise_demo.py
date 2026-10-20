"""
What does each noise type erase?  Per-scale signal-to-noise on YOUR images, for Gaussian vs simplex vs colored noise.

  python noise_demo.py --images a.jpg b.jpg c.jpg
  python noise_demo.py --data_dir path/to/clean_images --n 20 --simplex_wavelength 96 --simplex_persistence 0.75

SNR (dB) of structures of each size at noise level t (0 dB = signal equals noise). Where it is clearly below 0 dB the
structure is erased by the noise and the model must regenerate it; where it is clearly above 0 dB it survives
(and a missing weld / bent rod of that size will be kept by the reconstruction).
Also saves noise_demo.png (noise samples + their power spectra).
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stadiff.config import DEFAULT_ROI_BOX, make_cfg
from stadiff.data import ImgDS, list_images
from stadiff.diffusion import make_ab
from stadiff.noise import sample_noise

BANDS = [(1 / 128, 1 / 64), (1 / 64, 1 / 32), (1 / 32, 1 / 16), (1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1 / 2)]


def band_power(arr, H, W):
    """Mean orthonormal-FFT power per band. arr: (N,H,W)."""
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    P = np.abs(np.fft.fft2(arr, norm="ortho")) ** 2
    return np.array([P[:, (r >= lo) & (r < hi)].mean() for lo, hi in BANDS])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--H", type=int, default=192)
    p.add_argument("--W", type=int, default=256)
    p.add_argument("--norm", default="clahe", choices=["clahe", "none"])
    p.add_argument("--t", type=int, nargs="+", default=[300, 500, 800])
    p.add_argument("--simplex_wavelength", type=float, default=64.0)
    p.add_argument("--simplex_octaves", type=int, default=5)
    p.add_argument("--simplex_persistence", type=float, default=0.7)
    p.add_argument("--color_beta", type=float, default=1.0)
    p.add_argument("--out", default="noise_demo.png")
    a = p.parse_args()

    files = a.images or random.Random(0).sample(list_images(a.data_dir), min(a.n, len(list_images(a.data_dir))))
    cfg0 = make_cfg(a.H, a.W, 1, 64, 500, 300, DEFAULT_ROI_BOX, a.norm)
    ds = ImgDS(files, cfg0)
    x = torch.stack([ds[i] for i in range(len(files))])[:, 0].numpy().astype(np.float64)       # (N,H,W)
    P_sig = band_power(x, a.H, a.W)
    ab = make_ab("cpu")

    torch.manual_seed(0)
    like, m = torch.zeros(48, 1, a.H, a.W), torch.ones(48, 1, a.H, a.W)
    kinds, noise, psd = ["gaussian", "simplex", "colored"], {}, {}
    for k in kinds:
        cfg = make_cfg(a.H, a.W, 1, 64, 500, 300, DEFAULT_ROI_BOX, a.norm,
                       noise={"roi": k, "bg": k, "wavelength": a.simplex_wavelength, "octaves": a.simplex_octaves,
                              "persistence": a.simplex_persistence, "beta": a.color_beta})
        noise[k] = sample_noise(like, m, cfg)[:, 0].numpy().astype(np.float64)
        psd[k] = band_power(noise[k], a.H, a.W)

    print(f"{len(files)} images | SNR (dB) of structures of each size  (0 dB = signal equals noise)")
    for t in a.t:
        snr_t = float(ab[t] / (1 - ab[t]))
        print(f"\nnoise level t = {t}   (overall SNR per pixel {snr_t:.4f})")
        print(f"{'structure size':>16s} | " + " | ".join(f"{k:>8s}" for k in kinds))
        for b, (lo, hi) in enumerate(BANDS):
            row = [10 * np.log10(P_sig[b] * snr_t / psd[k][b]) for k in kinds]
            print(f"{int(1 / hi):>6d}-{int(1 / lo):<3d} px     | " + " | ".join(f"{v:8.1f}" for v in row))

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for i, k in enumerate(kinds):
        ax[i].imshow(noise[k][0], cmap="gray", vmin=-3, vmax=3)
        ax[i].set_title(f"{k} noise (unit variance)"); ax[i].axis("off")
    centers = [np.sqrt(lo * hi) for lo, hi in BANDS]
    for k in kinds:
        ax[3].plot(centers, psd[k], marker="o", label=k)
    ax[3].plot(centers, P_sig, "k--", marker="s", label="your images (signal)")
    ax[3].set_xscale("log"); ax[3].set_yscale("log"); ax[3].set_xlabel("spatial frequency (cycles/pixel)")
    ax[3].set_ylabel("power"); ax[3].set_title("power spectra"); ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(a.out, dpi=100)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
