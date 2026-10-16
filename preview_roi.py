"""
Check the hard-coded ROI rectangle on your images BEFORE training.
The rectangle must cover all three welds (and part of the rods) in EVERY image.

  python preview_roi.py --data_dir path/to/clean_images --n 12
  python preview_roi.py --data_dir path/to/clean_images --n 12 --roi_box 0.15 0.18 0.87 0.94
  python preview_roi.py --images a.jpg b.jpg c.jpg
Writes roi_preview.png (images are shown exactly as the model will see them: grayscale, resized, normalised).
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from stadiff.config import DEFAULT_ROI_BOX, make_cfg
from stadiff.data import ImgDS, list_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--n", type=int, default=12, help="how many random images to show from --data_dir")
    p.add_argument("--H", type=int, default=192)
    p.add_argument("--W", type=int, default=256)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--norm", default="clahe", choices=["clahe", "none"])
    p.add_argument("--roi_box", type=float, nargs=4, default=list(DEFAULT_ROI_BOX), metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="roi_preview.png")
    a = p.parse_args()

    if a.images:
        files = a.images
    else:
        allf = list_images(a.data_dir)
        files = random.Random(a.seed).sample(allf, min(a.n, len(allf)))
    cfg = make_cfg(a.H, a.W, a.channels, 64, 500, 300, a.roi_box, a.norm)
    ds = ImgDS(files, cfg)
    x0, y0, x1, y1 = a.roi_box

    cols = min(4, len(files))
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * a.H / a.W * rows + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, f in enumerate(files):
        ax = axes[i // cols, i % cols]
        img = ds[i]
        ax.imshow(((img[0] if a.channels == 1 else img.permute(1, 2, 0)).numpy() + 1) / 2,
                  cmap="gray" if a.channels == 1 else None, vmin=0, vmax=1)
        ax.add_patch(Rectangle((x0 * a.W, y0 * a.H), (x1 - x0) * a.W, (y1 - y0) * a.H, fill=False, ec="lime", lw=2))
        ax.set_title(Path(f).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(a.out, dpi=100)

    w0, h0 = Image.open(files[0]).size
    print(f"ROI box (fractions): x {x0:.2f}-{x1:.2f}, y {y0:.2f}-{y1:.2f}   = {100 * (x1 - x0) * (y1 - y0):.0f}% of the image")
    print(f"  at the model resolution {a.W}x{a.H}: x {x0 * a.W:.0f}-{x1 * a.W:.0f}, y {y0 * a.H:.0f}-{y1 * a.H:.0f}")
    print(f"  at the original resolution of {Path(files[0]).name} ({w0}x{h0}): x {x0 * w0:.0f}-{x1 * w0:.0f}, y {y0 * h0:.0f}-{y1 * h0:.0f}")
    print(f"saved {a.out} -> check that all welds are inside the green rectangle in every image")


if __name__ == "__main__":
    main()
