"""Dataset: a folder of (clean) images -> deterministic train / val / test splits -> PyTorch Dataset."""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
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
    """
    Loads the FULL image (no random crop / flip: the ROI rectangle is fixed in image coordinates),
    converts to grayscale (channels=1) or RGB (channels=3), resizes to (H, W), optionally normalises exposure
    with CLAHE, and returns a (C, H, W) tensor in [-1, 1].
    """

    def __init__(self, files, cfg):
        self.files, self.H, self.W = files, cfg["H"], cfg["W"]
        self.C, self.norm = cfg["channels"], cfg["norm"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mode = "L" if self.C == 1 else "RGB"
        try:
            img = Image.open(self.files[i])
            try:
                img.draft(mode, (self.W * 2, self.H * 2))          # fast JPEG downscale
            except Exception:
                pass
            img = img.convert(mode)
        except Exception as e:
            print(f"[warn] cannot read {self.files[i]}: {e}")
            return self[(i + 1) % len(self)]
        a = np.asarray(img.resize((self.W, self.H), Image.LANCZOS), np.float32) / 255.0
        if self.norm == "clahe":                                    # per-image exposure normalisation
            from skimage import exposure
            if a.ndim == 2:
                a = exposure.equalize_adapthist(a, clip_limit=0.02)
            else:
                a = np.stack([exposure.equalize_adapthist(a[..., c], clip_limit=0.02) for c in range(a.shape[-1])], -1)
        a = a.astype(np.float32) * 2 - 1
        a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a))


def write_demo_images(out_dir, n):
    """Write a small toy image folder (4:3 crops of a few sample photos). Only for smoke-testing the pipeline."""
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
    if not bases:
        for _ in range(6):
            arr = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
            bases.append(Image.fromarray(arr))
    for i in range(n):
        img = random.choice(bases)
        w, h = img.size
        cw = int(min(w, h * 4 / 3) * random.uniform(0.5, 1.0))
        ch = int(cw * 3 / 4)
        left, top = random.randint(0, w - cw), random.randint(0, h - ch)
        img.crop((left, top, left + cw, top + ch)).resize((320, 240), Image.LANCZOS).save(
            os.path.join(out_dir, f"img_{i:04d}.jpg"), quality=95)
    print(f"wrote {n} images to {out_dir}  (toy data: crops of {len(bases)} base images)")
