"""
Run a trained model on ONE image. The ROI is the fixed rectangle stored in the checkpoint.

  python infer.py --ckpt runs/exp1/best.pt --image photo.jpg
"""
import argparse

from stadiff.data import ImgDS
from stadiff.diffusion import make_ab
from stadiff.masks import fixed_mask
from stadiff.sampler import staged_sample
from stadiff.utils import get_device, load_model
from stadiff.visualize import save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--out", default="infer_result.png")
    a = p.parse_args()

    dev = get_device()
    model, cfg, _ = load_model(a.ckpt, dev)
    x0 = ImgDS([a.image], cfg)[0].unsqueeze(0).to(dev)
    m = fixed_mask(cfg, dev)

    rec, start = staged_sample(model, x0, m, make_ab(dev), cfg, a.sample_steps)
    ex = {"orig": x0.cpu(), "mask": m.cpu(), "corrupted": start.cpu(), "recon": rec.cpu(),
          "composite": (m * rec + (1 - m) * x0).cpu()}
    save_grid(ex, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
