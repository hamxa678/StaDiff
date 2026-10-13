"""Dataset: a folder of images -> deterministic train / val / test splits -> PyTorch Dataset."""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .config import IMG_EXTS


def list_images(data_dir):
    """All image files under data_dir (recursive)."""
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True)
        files += glob.glob(os.path.join(data_dir, "**", f"*{ext.upper()}"), recursive=True)
    return sorted(set(files))


def make_splits(data_dir, val_frac, test_frac, seed):
    """Deterministic split of image *files* (no image appears in two splits)."""
    files = list_images(data_dir)
    if len(files) < 3:
        raise SystemExit(f"Need at least 3 images in {data_dir}, found {len(files)}.")
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test, n_val = max(1, round(n * test_frac)), max(1, round(n * val_frac))
    return {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}


class ImgDS(Dataset):
    """Lazy loader. Train: random crop + flip. Val/test/infer: deterministic center crop. Output in [-1, 1]."""

    def __init__(self, files, size, train):
        self.files, self.size, self.train = files, size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i])
            try:
                img.draft("RGB", (self.size * 2, self.size * 2))   # fast JPEG downscale
            except Exception:
                pass
            img = img.convert("RGB")
        except Exception as e:
            print(f"[warn] cannot read {self.files[i]}: {e}")
            return self[(i + 1) % len(self)]
        w, h = img.size
        side = min(w, h)
        if self.train:
            side = int(side * random.uniform(0.6, 1.0))
            left, top = random.randint(0, w - side), random.randint(0, h - side)
        else:
            left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((self.size, self.size), Image.LANCZOS)
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def write_demo_images(out_dir, n):
    """Write a small toy image folder (random crops of a few sample photos) to try the pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    bases = []
    try:
        from skimage import data
        for name in ("astronaut", "coffee", "chelsea", "rocket"):
            try:
                bases.append(Image.fromarray(getattr(data, name)()).convert("RGB"))
            except Exception:
                pass
    except ImportError:
        pass
    if not bases:                                       # synthetic fallback
        for _ in range(6):
            s = 256
            y, x = np.mgrid[0:s, 0:s] / s
            c1, c2 = np.random.rand(3), np.random.rand(3)
            im = Image.fromarray((np.clip(x[..., None] * c1 + y[..., None] * c2, 0, 1) * 255).astype(np.uint8))
            d = ImageDraw.Draw(im)
            for _ in range(10):
                rr, cx, cy = random.randint(10, 50), random.randint(0, s), random.randint(0, s)
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=tuple(random.randint(0, 255) for _ in range(3)))
            bases.append(im)
    for i in range(n):
        img = random.choice(bases)
        w, h = img.size
        side = int(min(w, h) * random.uniform(0.4, 1.0))
        left, top = random.randint(0, w - side), random.randint(0, h - side)
        img = img.crop((left, top, left + side, top + side)).resize((128, 128), Image.LANCZOS)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img.save(os.path.join(out_dir, f"img_{i:04d}.jpg"), quality=95)
    print(f"wrote {n} images to {out_dir}  (toy data: crops of {len(bases)} base images)")
