"""Validation / test loop: eps-prediction error per region and full two-phase reconstruction quality."""
from collections import defaultdict

import numpy as np
import torch

from .diffusion import build_t_map, net_input, q_sample
from .masks import random_masks
from .metrics import region_psnr, ssim, to01
from .sampler import staged_sample


@torch.no_grad()
def evaluate(model, loader, cfg, ab, dev, n_steps=50, seed=0, max_batches=None, n_examples=0, collect_rows=False):
    """
    Fixed seed -> fixed masks and noise, so numbers are comparable between runs and checkpoints.
    Returns (summary dict, per-image rows, example tensors for plotting).
    """
    model.eval()
    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    buckets = {"early": (1, t_bg), "late": (t_bg + 1, t_roi)}        # s <= t_bg   /   s > t_bg
    acc, meters, rows, ex = defaultdict(float), defaultdict(list), [], defaultdict(list)
    gen = torch.Generator().manual_seed(seed)
    devices = [0] if str(dev).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for bi, x0 in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x0 = x0.to(dev)
            B, _, H, W = x0.shape
            m = random_masks(B, H, W, dev, gen)

            # (A) eps-prediction error per region in the two phases
            for name, (lo, hi) in buckets.items():
                s = torch.randint(lo, hi + 1, (B,), device=dev)
                t_map = build_t_map(s, m, t_roi, t_bg)
                x_t, eps = q_sample(x0, t_map, ab)
                err = ((model(net_input(x_t, m, t_map, t_roi), s) - eps) ** 2).mean(1, keepdim=True)
                acc[f"{name}_roi"] += (err * m).sum().item()
                acc[f"{name}_roi_n"] += m.sum().item()
                acc[f"{name}_bg"] += (err * (1 - m)).sum().item()
                acc[f"{name}_bg_n"] += (1 - m).sum().item()

            # (B) full two-phase reconstruction
            rec, start = staged_sample(model, x0, m, ab, cfg, n_steps)
            comp = m * rec + (1 - m) * x0                                 # clean background pasted back
            roi, bg, full = (m > 0.5).float(), (m < 0.05).float(), torch.ones_like(m)
            bm = {"roi_frac": roi.mean((1, 2, 3))}
            for name, img in (("corrupted", start), ("recon", rec), ("composite", comp)):
                a, b = to01(img), to01(x0)
                bm[f"{name}/psnr_full"] = region_psnr(a, b, full)[0]
                bm[f"{name}/ssim"] = ssim(a, b)
                if name != "composite":
                    for rn, rg in (("roi", roi), ("bg", bg)):
                        p, v = region_psnr(a, b, rg)
                        bm[f"{name}/psnr_{rn}"] = torch.where(v, p, torch.full_like(p, float("nan")))
            for k, v in bm.items():
                meters[k].append(v.cpu())
            if collect_rows:
                for j in range(B):
                    rows.append({"idx": bi * loader.batch_size + j, **{k: float(v[j]) for k, v in bm.items()}})
            if sum(len(v) for v in ex["orig"]) < n_examples:
                for k, v in (("orig", x0), ("mask", m), ("corrupted", start), ("recon", rec), ("composite", comp)):
                    ex[k].append(v.cpu())

    sm = {k: torch.cat(v).nanmean().item() for k, v in meters.items()}
    tot = lambda n: (acc[f"{n}_roi"] + acc[f"{n}_bg"]) / max(acc[f"{n}_roi_n"] + acc[f"{n}_bg_n"], 1)
    for n in buckets:
        sm[f"eps_mse_{n}_roi"] = acc[f"{n}_roi"] / max(acc[f"{n}_roi_n"], 1)
        sm[f"eps_mse_{n}_bg"] = acc[f"{n}_bg"] / max(acc[f"{n}_bg_n"], 1)
    sm["val_eps_mse"] = float(np.mean([tot(n) for n in buckets]))        # used for best-checkpoint selection
    examples = {k: torch.cat(v)[:n_examples] for k, v in ex.items()} if n_examples else {}
    model.train()
    return sm, rows, examples


def print_summary(sm, ind="    "):
    print(f"{ind}eps-MSE  s<=t_bg (ROI/BG): {sm['eps_mse_early_roi']:.4f}/{sm['eps_mse_early_bg']:.4f}   "
          f"s>t_bg (ROI/BG): {sm['eps_mse_late_roi']:.4f}/{sm['eps_mse_late_bg']:.4f}   total: {sm['val_eps_mse']:.4f}")
    for name in ("corrupted", "recon", "composite"):
        s = f"{ind}{name:<10s} PSNR full {sm[f'{name}/psnr_full']:.2f}"
        if name != "composite":
            s += f" | ROI {sm[f'{name}/psnr_roi']:.2f} | BG {sm[f'{name}/psnr_bg']:.2f}"
        print(s + f" | SSIM {sm[f'{name}/ssim']:.3f}")
