"""
Run a trained model on ONE image with your own ROI box (fractions of the image: x0 y0 x1 y1).

  python infer.py --ckpt runs/exp1/best.pt --image photo.jpg --box 0.4 0.4 0.8 0.8
  python infer.py --ckpt runs/exp1/best.pt --image photo.jpg            # random ROI
"""
import argparse

import torch

from stadiff.data import ImgDS
from stadiff.diffusion import make_ab
from stadiff.masks import box_mask, random_masks
from stadiff.sampler import staged_sample
from stadiff.utils import get_device, load_model
from stadiff.visualize import save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="infer_result.png")
    a = p.parse_args()

    dev = get_device()
    model, cfg, _ = load_model(a.ckpt, dev)
    S = cfg["size"]
    x0 = ImgDS([a.image], S, False)[0].unsqueeze(0).to(dev)
    m = box_mask(a.box, S, S, dev) if a.box else random_masks(1, S, S, dev, torch.Generator().manual_seed(a.seed))

    rec, start = staged_sample(model, x0, m, make_ab(dev), cfg, a.sample_steps)
    ex = {"orig": x0.cpu(), "mask": m.cpu(), "corrupted": start.cpu(), "recon": rec.cpu(),
          "composite": (m * rec + (1 - m) * x0).cpu()}
    save_grid(ex, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
