"""
Test a trained StaDiff on ANOMALOUS images.

  python anomaly_test.py --ckpt runs/exp2/best.pt --anomaly_dir "path/to/anomalous_images"
  python anomaly_test.py --ckpt runs/exp2/best.pt --anomaly_dir ... --mask_dir "path/to/defect_masks"   # + pixel AUROC

How it works: each image goes through the same ROI regeneration used in training (ROI noised to t_roi, background to
t_bg, staged reverse process). The anomaly map is |input - reconstruction| (averaged over --repeats noise draws, then
smoothed). Image score = max / mean of top-k% / mean of that map inside the ROI. Anomalous images should score higher.

Normal images: by default the held-out clean TEST split of the run (runs/exp2/splits.json); --normal_dir overrides.
The decision threshold is calibrated on the clean VALIDATION split (--thr_pct percentile of its scores).
Writes to anomaly_results/ (or --out): scores.csv, metrics.json, roc.png, score_hist.png, examples.png
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from stadiff.anomaly import load_gt_masks, pixel_auroc, score_files
from stadiff.data import list_images
from stadiff.diffusion import make_ab
from stadiff.masks import fixed_mask
from stadiff.utils import get_device, load_model
from stadiff.visualize import save_anomaly_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--anomaly_dir", required=True, help="folder with anomalous images")
    p.add_argument("--normal_dir", default=None, help="default: clean test split from splits.json next to the checkpoint")
    p.add_argument("--mask_dir", default=None, help="optional ground-truth defect masks (matched by file name)")
    p.add_argument("--out", default="anomaly_results")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sample_steps", type=int, default=50, help="reverse steps (more = slower, maybe better)")
    p.add_argument("--repeats", type=int, default=3, help="noise draws averaged per image")
    p.add_argument("--sigma", type=float, default=None, help="map smoothing in pixels (default H/96)")
    p.add_argument("--topk", type=float, default=0.01, help="fraction of ROI pixels for the 'topk' score")
    p.add_argument("--score", default="topk", choices=["max", "topk", "mean"], help="score used for threshold/plots")
    p.add_argument("--region", default="roi", choices=["roi", "all"], help="score only inside the ROI, or the whole image")
    p.add_argument("--thr_pct", type=float, default=99.0, help="threshold = this percentile of clean validation scores")
    p.add_argument("--n_examples", type=int, default=4, help="example rows per class in examples.png")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = get_device()
    model, cfg, ck = load_model(a.ckpt, dev)
    ab = make_ab(dev)
    sigma = a.sigma if a.sigma is not None else cfg["H"] / 96
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split_path = Path(a.ckpt).parent / "splits.json"
    splits = json.loads(split_path.read_text()) if split_path.exists() else None

    anom_files = list_images(a.anomaly_dir)
    if a.normal_dir:
        normal_files = list_images(a.normal_dir)
    elif splits:
        normal_files = splits["test"]
    else:
        raise SystemExit("No splits.json next to the checkpoint: pass --normal_dir with held-out clean images.")
    if not anom_files or not normal_files:
        raise SystemExit(f"Found {len(normal_files)} normal and {len(anom_files)} anomalous images - need both.")
    print(f"model trained {ck['step']} steps | {len(normal_files)} normal | {len(anom_files)} anomalous | cfg {cfg}")

    torch.manual_seed(a.seed)
    kw = dict(batch=a.batch, n_steps=a.sample_steps, n_repeats=a.repeats, sigma=sigma, topk=a.topk, region=a.region)
    print("scoring normal test images:")
    sc_n, maps_n, recs_n, normal_files = score_files(normal_files, model, cfg, ab, dev, keep_maps=True,
                                                     cache_file=out / "cache_normal.npz", tag="normal", **kw)
    print("scoring anomalous images:")
    sc_a, maps_a, recs_a, anom_files = score_files(anom_files, model, cfg, ab, dev, keep_maps=True,
                                                   cache_file=out / "cache_anomalous.npz", tag="anomalous", **kw)
    if splits and splits.get("val"):
        print("scoring clean validation images (threshold calibration):")
        ref = score_files(splits["val"], model, cfg, ab, dev, cache_file=out / "cache_val.npz", tag="val", **kw)[0][a.score]
        calib = "clean validation split"
    else:
        ref, calib = sc_n[a.score], "the normal test images themselves (optimistic!)"
        print("[warn] no validation split found: threshold calibrated on the normal test images")
    thr = float(np.percentile(ref, a.thr_pct))

    # ---- image-level metrics ----------------------------------------------------------------
    y = np.r_[np.zeros(len(normal_files)), np.ones(len(anom_files))]
    res = {"n_normal": len(normal_files), "n_anomalous": len(anom_files), "region": a.region, "score_used": a.score,
           "repeats": a.repeats, "sample_steps": a.sample_steps}
    for k in ("max", "topk", "mean"):
        res[f"image_auroc_{k}"] = float(roc_auc_score(y, np.r_[sc_n[k], sc_a[k]]))
    s_all = np.r_[sc_n[a.score], sc_a[a.score]]
    pred = s_all > thr
    tpr, fpr = float(pred[y == 1].mean()), float(pred[y == 0].mean())
    prec = float(y[pred].mean()) if pred.any() else float("nan")
    res.update({"threshold": thr, "threshold_calibrated_on": calib, "threshold_percentile": a.thr_pct,
                "TPR_anomalous_detected": tpr, "FPR_normal_flagged": fpr, "precision": prec})
    res["median_score_normal"] = float(np.median(sc_n[a.score]))
    res["median_score_anomalous"] = float(np.median(sc_a[a.score]))

    # ---- pixel-level AUROC (optional) --------------------------------------------------------
    region_hw = (fixed_mask(cfg, "cpu")[0, 0] > 0.5) if a.region == "roi" else torch.ones(cfg["H"], cfg["W"], dtype=torch.bool)
    gts_a = has = None
    if a.mask_dir:
        gts_a, has = load_gt_masks(anom_files, a.mask_dir, cfg)
        print(f"ground-truth masks found for {int(has.sum())}/{len(anom_files)} anomalous images")
        if has.any():
            maps = torch.cat([maps_n, maps_a[torch.from_numpy(has)]])
            gts = torch.cat([torch.zeros(len(normal_files), cfg["H"], cfg["W"], dtype=torch.bool), gts_a[torch.from_numpy(has)]])
            res["pixel_auroc"] = pixel_auroc(maps, gts, region_hw)
            res["pixel_auroc_n_anomalous_with_mask"] = int(has.sum())

    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "score_max", "score_topk", "score_mean", f"flagged(score_{a.score}>thr)"])
        for lab, files, sc in (("normal", normal_files, sc_n), ("anomalous", anom_files, sc_a)):
            for i, fn in enumerate(files):
                w.writerow([fn, lab, sc["max"][i], sc["topk"][i], sc["mean"][i], int(sc[a.score][i] > thr)])

    # ---- figures -------------------------------------------------------------------------------
    plt.figure(figsize=(5.2, 5))
    for k in ("max", "topk", "mean"):
        fp, tp, _ = roc_curve(y, np.r_[sc_n[k], sc_a[k]])
        plt.plot(fp, tp, lw=2, label=f"{k}  (AUROC {res[f'image_auroc_{k}']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("false positive rate (normal flagged)"); plt.ylabel("true positive rate")
    plt.title("Image-level ROC"); plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=110); plt.close()

    plt.figure(figsize=(7, 4))
    bins = np.linspace(min(sc_n[a.score].min(), sc_a[a.score].min()), max(sc_n[a.score].max(), sc_a[a.score].max()), 30)
    plt.hist(sc_n[a.score], bins=bins, alpha=0.6, label=f"normal (n={len(normal_files)})", color="tab:blue")
    plt.hist(sc_a[a.score], bins=bins, alpha=0.6, label=f"anomalous (n={len(anom_files)})", color="tab:red")
    plt.axvline(thr, color="k", ls="--", label=f"threshold ({a.thr_pct:g}th pct of clean val)")
    plt.xlabel(f"image score ({a.score})"); plt.ylabel("images"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "score_hist.png", dpi=110); plt.close()

    from stadiff.data import ImgDS
    rng = random.Random(a.seed)
    pick = lambda n, k: rng.sample(range(n), min(k, n))
    vmax = float(np.percentile(maps_a.float().numpy().ravel()[::37], 99.5))
    rows = []
    for lab, files, sc, maps, recs, gts_ in (("anomalous", anom_files, sc_a, maps_a, recs_a, gts_a),
                                             ("normal", normal_files, sc_n, maps_n, recs_n, None)):
        ds = ImgDS(files, cfg, out / f"cache_{lab}.npz", lab)
        for i in pick(len(files), a.n_examples):
            gt = gts_[i] if (gts_ is not None and has[i]) else None
            rows.append({"x0": ds[i], "rec": recs[i].float(), "amap": maps[i], "gt": gt,
                         "title": f"{lab}: {Path(files[i]).name}  score {sc[a.score][i]:.4f}  ({'FLAGGED' if sc[a.score][i] > thr else 'not flagged'})"})
    save_anomaly_examples(rows, fixed_mask(cfg, "cpu"), out / "examples.png", vmax)

    print("\n================ anomaly test ================")
    for k in ("max", "topk", "mean"):
        print(f"  image AUROC ({k:<4s}): {res[f'image_auroc_{k}']:.3f}")
    if "pixel_auroc" in res and res["pixel_auroc"] is not None:
        print(f"  pixel AUROC          : {res['pixel_auroc']:.3f}   ({res['pixel_auroc_n_anomalous_with_mask']} anomalous images with masks)")
    print(f"  threshold {thr:.4f} (from {calib}): detects {100 * tpr:.1f}% of anomalous images, flags {100 * fpr:.1f}% of normal images")
    print(f"  median score: normal {res['median_score_normal']:.4f} | anomalous {res['median_score_anomalous']:.4f}")
    print(f"saved scores.csv, metrics.json, roc.png, score_hist.png, examples.png to {out}/")


if __name__ == "__main__":
    main()
