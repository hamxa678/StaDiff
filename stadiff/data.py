"""Dataset: a folder of (clean) images -> deterministic train / val / test splits -> in-memory PyTorch Dataset."""
import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

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


def _load_one(path, cfg):
    """One image -> uint8 array (C,H,W): grayscale/RGB, resized to (H,W), optional CLAHE. None if unreadable."""
    H, W, C = cfg["H"], cfg["W"], cfg["channels"]
    mode = "L" if C == 1 else "RGB"
    try:
        img = Image.open(path)
        try:
            img.draft(mode, (W * 2, H * 2))                       # fast JPEG downscale
        except Exception:
            pass
        a = np.asarray(img.convert(mode).resize((W, H), Image.LANCZOS), np.float32) / 255.0
        if cfg["norm"] == "clahe":                                # per-image exposure normalisation
            from skimage import exposure
            if a.ndim == 2:
                a = exposure.equalize_adapthist(a, clip_limit=0.02)
            else:
                a = np.stack([exposure.equalize_adapthist(a[..., c], clip_limit=0.02) for c in range(C)], -1)
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        return a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    except Exception as e:
        print(f"[warn] cannot read {path}: {e}", flush=True)
        return None


def load_tensors(files, cfg, cache_file=None, workers=8, tag="images"):
    """
    Read + preprocess ALL images once (threaded, with progress) and keep them in memory as uint8.
    With cache_file the result is stored on local disk, so slow storage (e.g. Google Drive) is only read once.
    """
    key = json.dumps({"files": list(files), "H": cfg["H"], "W": cfg["W"], "C": cfg["channels"], "norm": cfg["norm"]})
    if cache_file and os.path.exists(cache_file):
        try:
            z = np.load(cache_file, allow_pickle=False)
            if str(z["key"]) == key:
                print(f"[data] {tag}: {len(files)} images loaded from cache {cache_file}", flush=True)
                return z["data"], list(files)
        except Exception:
            pass
    print(f"[data] {tag}: reading {len(files)} images (first time only; cached afterwards)...", flush=True)
    t0, out = time.time(), [None] * len(files)
    with ThreadPoolExecutor(workers) as ex:
        for i, arr in enumerate(ex.map(lambda f: _load_one(f, cfg), files)):
            out[i] = arr
            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print(f"[data] {tag}: {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    ok = [i for i, a in enumerate(out) if a is not None]
    if not ok:
        raise SystemExit(f"None of the {tag} images could be read.")
    data, kept = np.stack([out[i] for i in ok]), [files[i] for i in ok]
    if cache_file and len(ok) == len(files):
        np.savez(cache_file, data=data, key=key)
    return data, kept


class ImgDS(Dataset):
    """
    In-memory dataset. Loads the FULL image (no random crop / flip: the ROI rectangle is fixed in image
    coordinates). Returns a (C, H, W) float tensor in [-1, 1].
    """

    def __init__(self, files, cfg, cache_file=None, tag="images"):
        self.data, self.files = load_tensors(files, cfg, cache_file, tag=tag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i]).float() / 127.5 - 1.0


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
