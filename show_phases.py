"""
Illustrate the two phases of the reverse process and PROVE that the background is not changed in Phase 1.

  python show_phases.py --ckpt runs/exp1/best.pt                       # first test image, random ROI
  python show_phases.py --ckpt runs/exp1/best.pt --image photo.jpg --gif

Writes to phase_figures/ (or --out_dir):
  phases_filmstrip.png   image / change-vs-start heatmap / noise-level map at several steps s
  phases_curves.png      noise level, change vs start, and PSNR per region over the reverse process
  phases.gif             animation for slides (with --gif)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stadiff.data import ImgDS
from stadiff.diffusion import build_t_map, make_ab
from stadiff.masks import fixed_mask
from stadiff.metrics import region_psnr, to01
from stadiff.sampler import staged_sample
from stadiff.utils import get_device, load_model
from stadiff.visualize import save_filmstrip, save_gif, save_phase_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", default=None, help="default: a test image from splits.json next to the checkpoint")
    p.add_argument("--index", type=int, default=0, help="which test image to use when --image is not given")
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--out_dir", default="phase_figures")
    a = p.parse_args()

    dev = get_device()
    model, cfg, _ = load_model(a.ckpt, dev)
    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    ab = make_ab(dev)
    if a.image is None:
        a.image = json.loads((Path(a.ckpt).parent / "splits.json").read_text())["test"][a.index]
    print(f"image: {a.image}")
    x0 = ImgDS([a.image], cfg)[0].unsqueeze(0).to(dev)
    m = fixed_mask(cfg, dev)

    torch.manual_seed(a.seed)
    rec, x_start, traj = staged_sample(model, x0, m, ab, cfg, a.sample_steps, return_traj=True)

    # --- three regions -----------------------------------------------------------------------
    roi = (m > 0.5).float()
    fixed = (m * (t_roi - t_bg) < 0.5).float()          # timestep never moves in Phase 1 -> must stay untouched
    band = ((roi + fixed) == 0).float()                 # blurred ROI border: partly denoised in Phase 1
    regs = {"ROI (core)": roi, "ROI border band": band, "Background (fixed in Phase 1)": fixed}
    print({k: int(v.sum().item()) for k, v in regs.items()}, "pixels per region")

    # --- curves --------------------------------------------------------------------------------
    s_list = [s for s, _ in traj]
    curves = {"level": {k: [] for k in regs}, "change": {k: [] for k in regs}, "psnr": {k: [] for k in regs}}
    for s, x in traj:
        t_map = build_t_map(torch.full((1,), int(s), device=dev), m, t_roi, t_bg).float()
        for k, reg in regs.items():
            n = reg.sum().clamp(min=1)
            curves["level"][k].append(((t_map * reg).sum() / n).item())
            curves["change"][k].append((((x - x_start).abs().mean(1, keepdim=True) * reg).sum() / n).item())
            curves["psnr"][k].append(region_psnr(to01(x), to01(x0), reg)[0].item())

    # --- the proof, as a number ----------------------------------------------------------------
    worst = max((x - x_start).abs().mul(fixed).max().item() for s, x in traj if s >= t_bg)
    print(f"\nMax change of the fixed background pixels during Phase 1 (s from {t_roi} to {t_bg}): {worst:.8f}")
    roi_change = max((x - x_start).abs().mul(roi).max().item() for s, x in traj if s >= t_bg)
    print(f"Max change of the ROI core pixels during Phase 1:                          {roi_change:.8f}")

    # --- figures --------------------------------------------------------------------------------
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = list(np.linspace(t_roi, t_bg, 4)) + list(np.linspace(t_bg, 0, 4)[1:])
    save_filmstrip(traj, x_start, m, cfg, targets, out / "phases_filmstrip.png")
    save_phase_curves(s_list, curves, cfg, out / "phases_curves.png")
    if a.gif:
        save_gif(traj, x_start, m, cfg, out / "phases.gif")
    print(f"saved figures to {out}/")


if __name__ == "__main__":
    main()
