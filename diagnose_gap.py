"""
Why is the ROI reconstructed worse than the background?  Measure the causes on the held-out clean images.

  python diagnose_gap.py --ckpt runs/exp2/best.pt

Three runs per image (PSNR vs the clean original, ROI core vs background):
  A  staged            ROI noised to t_roi, background to t_bg   (the normal StaDiff run)
  B  uniform t_bg      the WHOLE image noised to t_bg only, then denoised together (no extra ROI noise)
  C  staged, mean of K the average of K staged reconstructions (removes the randomness of a single sample)
Reading the table:
  B ROI - A ROI   = what the extra noise (t_bg -> t_roi) cost in the ROI   (information destroyed)
  B BG  - B ROI   = ROI vs background gap at IDENTICAL noise and SNR       (content difficulty)
  C ROI - A ROI   = what one random sample costs vs the average estimate   (sampling randomness)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stadiff.data import ImgDS
from stadiff.diffusion import make_ab
from stadiff.masks import fixed_mask
from stadiff.metrics import region_psnr, to01
from stadiff.sampler import staged_sample
from stadiff.utils import get_device, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--n", type=int, default=64, help="max number of images")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--K", type=int, default=4, help="samples averaged in run C")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = get_device()
    model, cfg, ck = load_model(a.ckpt, dev)
    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    ab = make_ab(dev)
    files = json.loads((Path(a.ckpt).parent / "splits.json").read_text())[a.split][: a.n]
    dl = DataLoader(ImgDS(files, cfg), batch_size=a.batch, shuffle=False)
    m1 = fixed_mask(cfg, dev)
    roi1, bg1 = (m1 > 0.5).float(), (m1 < 0.05).float()
    snr = lambda t: float(ab[t] / (1 - ab[t]))
    print(f"{len(files)} {a.split} images | noise level t={t_bg}: SNR {snr(t_bg):.3f} ({10 * np.log10(snr(t_bg)):.1f} dB)"
          f"  vs  t={t_roi}: SNR {snr(t_roi):.3f} ({10 * np.log10(snr(t_roi)):.1f} dB)")

    torch.manual_seed(a.seed)
    acc = {k: [] for k in ["A_in_roi", "A_in_bg", "B_in_roi", "B_in_bg", "A_roi", "A_bg", "B_roi", "B_bg", "C_roi", "C_bg"]}
    steps_b = max(2, round(a.sample_steps * t_bg / t_roi))             # same step size as run A
    for x0 in dl:
        x0 = x0.to(dev)
        B = x0.shape[0]
        m, roi, bg = m1.expand(B, -1, -1, -1), roi1.expand(B, -1, -1, -1), bg1.expand(B, -1, -1, -1)
        t01 = to01(x0)

        def psnr(img, reg):
            v, ok = region_psnr(to01(img), t01, reg)
            return v[ok].cpu()

        recA, inA = staged_sample(model, x0, m, ab, cfg, a.sample_steps)
        recB, inB = staged_sample(model, x0, m, ab, cfg, steps_b, s_start=t_bg)
        recC = torch.stack([staged_sample(model, x0, m, ab, cfg, a.sample_steps)[0] for _ in range(a.K)]).mean(0)
        for name, img in (("A_in", inA), ("B_in", inB), ("A", recA), ("B", recB), ("C", recC)):
            acc[f"{name}_roi"].append(psnr(img, roi)); acc[f"{name}_bg"].append(psnr(img, bg))

    r = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    print("\nPSNR vs the clean original (dB)             ROI      background")
    print(f"  input after corruption, run A (staged)  {r['A_in_roi']:7.2f}   {r['A_in_bg']:7.2f}")
    print(f"  input after corruption, run B (uniform) {r['B_in_roi']:7.2f}   {r['B_in_bg']:7.2f}")
    print(f"  A  staged reconstruction                {r['A_roi']:7.2f}   {r['A_bg']:7.2f}")
    print(f"  B  uniform-{t_bg} reconstruction          {r['B_roi']:7.2f}   {r['B_bg']:7.2f}")
    print(f"  C  staged, mean of {a.K} samples            {r['C_roi']:7.2f}   {r['C_bg']:7.2f}")
    print("\nWhere the ROI gap comes from:")
    print(f"  total gap in run A (background - ROI)                  : {r['A_bg'] - r['A_roi']:6.2f} dB")
    print(f"  content difficulty (gap at identical noise, run B)     : {r['B_bg'] - r['B_roi']:6.2f} dB")
    print(f"  extra noise in the ROI (B ROI - A ROI)                 : {r['B_roi'] - r['A_roi']:6.2f} dB")
    print(f"  one random sample vs average of {a.K} (C ROI - A ROI)     : {r['C_roi'] - r['A_roi']:6.2f} dB")


if __name__ == "__main__":
    main()
