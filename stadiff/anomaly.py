"""Anomaly scoring: reconstruction error inside the ROI -> anomaly map -> image-level scores -> metrics."""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import ImgDS, list_images
from .masks import fixed_mask, gaussian_blur
from .sampler import staged_sample


@torch.no_grad()
def anomaly_maps(model, x0, cfg, ab, n_steps=50, n_repeats=3, sigma=2.0):
    """
    x0: (B,C,H,W). Regenerates the ROI (staged reverse process) n_repeats times and averages |x0 - recon|.
    Returns (smoothed anomaly map (B,1,H,W), reconstruction of the last repeat).
    """
    B = x0.shape[0]
    m = fixed_mask(cfg, x0.device).expand(B, -1, -1, -1)
    acc, rec = 0.0, None
    for _ in range(n_repeats):
        rec, _ = staged_sample(model, x0, m, ab, cfg, n_steps)
        acc = acc + (rec - x0).abs().mean(1, keepdim=True)
    return gaussian_blur(acc / n_repeats, sigma), rec


def image_scores(amap, region, topk=0.01):
    """amap (B,1,H,W), region (1,1,H,W). Per image: max / mean of the top-k fraction / mean of the map inside the region."""
    out = {"max": [], "topk": [], "mean": []}
    reg = region[0] > 0.5                                         # (1,H,W) bool
    for a in amap:
        v = a[reg]
        k = max(1, int(topk * v.numel()))
        out["max"].append(v.max())
        out["topk"].append(v.topk(k).values.mean())
        out["mean"].append(v.mean())
    return {k: torch.stack(v).cpu().numpy() for k, v in out.items()}


def score_files(files, model, cfg, ab, dev, batch=16, n_steps=50, n_repeats=3, sigma=2.0, topk=0.01,
                region="roi", keep_maps=False, cache_file=None, tag="images"):
    """Score a list of image files (unreadable ones are skipped).
    Returns (scores dict of (N,) arrays, maps (N,1,H,W) fp16 or None, recs or None, list of files actually scored)."""
    ds = ImgDS(files, cfg, cache_file, tag)                       # reads every image once; cache_file avoids re-reading
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    if region == "roi":
        reg = (fixed_mask(cfg, dev) > 0.5).float()
    else:
        reg = torch.ones(1, 1, cfg["H"], cfg["W"], device=dev)
    scores = {k: [] for k in ("max", "topk", "mean")}
    maps, recs = [], []
    for i, x0 in enumerate(dl):
        x0 = x0.to(dev)
        amap, rec = anomaly_maps(model, x0, cfg, ab, n_steps, n_repeats, sigma)
        s = image_scores(amap, reg, topk)
        for k in scores:
            scores[k].append(s[k])
        if keep_maps:
            maps.append(amap.half().cpu())
            recs.append(rec.half().cpu())
        print(f"\r  scored {min((i + 1) * batch, len(ds))}/{len(ds)}", end="", flush=True)
    print()
    return ({k: np.concatenate(v) for k, v in scores.items()},
            torch.cat(maps) if keep_maps else None, torch.cat(recs) if keep_maps else None, ds.files)


def load_gt_masks(img_files, mask_dir, cfg):
    """Ground-truth masks matched to images by file stem (also tries stem_mask / stem_gt / stem-mask).
    Returns (bool tensor (N,H,W), has_mask array (N,))."""
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(mask_dir)}
    H, W = cfg["H"], cfg["W"]
    gts, has = [], []
    for f in img_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        path = next((by_stem[c] for c in (stem, f"{stem}_mask", f"{stem}_gt", f"{stem}-mask") if c in by_stem), None)
        if path is None:
            gts.append(torch.zeros(H, W, dtype=torch.bool)); has.append(False)
        else:
            m = np.asarray(Image.open(path).convert("L").resize((W, H), Image.NEAREST)) > 127
            gts.append(torch.from_numpy(m)); has.append(True)
    return torch.stack(gts), np.array(has)


def pixel_auroc(maps, gts, region, max_pixels=2_000_000, seed=0):
    """Pooled pixel-level AUROC. maps (N,1,H,W), gts (N,H,W) bool, region (H,W) bool. None if only one class."""
    from sklearn.metrics import roc_auc_score
    s = maps[:, 0][:, region].float().numpy().ravel()
    y = gts[:, region].numpy().ravel()
    if y.min() == y.max():
        return None
    if s.size > max_pixels:
        idx = np.random.RandomState(seed).choice(s.size, max_pixels, replace=False)
        s, y = s[idx], y[idx]
    return float(roc_auc_score(y, s))
