# StaDiff - Staged Diffusion with a fixed ROI rectangle (grayscale, your own clean images)

The ROI (one hard-coded rectangle covering all welds and part of the rods) is noised up to `t_roi` (e.g. 500),
the background only up to `t_bg` (e.g. 300). The reverse process is staged: Phase 1 denoises only the ROI
until it reaches the background's level, Phase 2 denoises the whole image together.

## Quick start

```bash
pip install -r requirements.txt

# 1. Check the hard-coded ROI rectangle on your clean images (it must contain all welds in EVERY image)
python preview_roi.py --data_dir path/to/clean_images --n 12
python preview_roi.py --data_dir path/to/clean_images --n 12 --roi_box 0.15 0.18 0.87 0.94   # try another box

# 2. Train (grayscale, 192x256, exposure-normalised, ROI box as above)
python train.py --data_dir path/to/clean_images --out runs/exp1 --t_roi 500 --t_bg 300 --steps 20000 --amp   # progress prints from step 1; data is read once and cached

# 3. Evaluate on the held-out test split, one image, and phase illustrations
python test.py        --ckpt runs/exp1/best.pt
python infer.py       --ckpt runs/exp1/best.pt --image photo.jpg
python show_phases.py --ckpt runs/exp1/best.pt --gif
```

`--roi_box x0 y0 x1 y1` are FRACTIONS of the image (origin top-left), so they do not depend on the resolution.
The default is `0.15 0.18 0.87 0.94` (about 55% of the image); it is stored in `stadiff/config.py`
(`DEFAULT_ROI_BOX`) and inside every checkpoint, so test/infer use the same rectangle automatically.

## Testing on anomalous images

```bash
python anomaly_test.py --ckpt runs/exp2/best.pt --anomaly_dir "path/to/anomalous_images"
python anomaly_test.py --ckpt runs/exp2/best.pt --anomaly_dir "path/to/anomalous_images" --mask_dir "path/to/defect_masks"
```

- Clean images for the comparison = the held-out TEST split of the run (`runs/exp2/splits.json`), never seen in training.
  Use `--normal_dir` to give another folder of clean images.
- Every image goes through the same ROI regeneration used in training; the anomaly map is |input - reconstruction|
  (averaged over `--repeats` noise draws, smoothed). Image score = `max`, `topk` (mean of the top 1% pixels) or `mean`
  of that map inside the ROI. Anomalous images should get higher scores.
- The decision threshold is calibrated on the clean VALIDATION split (`--thr_pct`, default 99th percentile).
- `--mask_dir` (optional): ground-truth defect masks matched to images by file name (`img.png` or `img_mask.png`),
  adds pixel-level AUROC.
- Output in `anomaly_results/`: `metrics.json`, `scores.csv`, `roc.png`, `score_hist.png`, `examples.png`.
- Defects outside the ROI rectangle cannot be detected (`--region all` scores the whole image instead).
- Choose the score variant (`max`/`topk`/`mean`) and `--thr_pct` on validation data, not on the test results.

## What changed compared with the random-mask version

- Images are loaded from your folder as grayscale (`--channels 3` for RGB), resized to `--H x --W` (default 192x256,
  must be divisible by 4), and exposure-normalised with CLAHE (`--norm none` to disable).
- No random crop / flip: the ROI rectangle is fixed in image coordinates, so the geometry must not move.
- No random masks: every training / validation / test step uses the same soft rectangle. (`random_masks` is still in
  `stadiff/masks.py`, unused, in case you want it back.)
- Use ONLY clean (anomaly-free) images in `--data_dir`.

## Project layout

```
stadiff_project/
├── preview_roi.py        # check the ROI rectangle on your images
├── train.py              # train + validate, saves best.pt / last.pt
├── test.py               # evaluate reconstruction quality on the held-out clean test split
├── anomaly_test.py       # score CLEAN vs ANOMALOUS images: image/pixel AUROC, threshold, examples
├── infer.py              # run on one image
├── show_phases.py        # figures/GIF that illustrate Phase 1 vs Phase 2
├── make_demo_data.py     # toy dataset, only for smoke-testing the code
├── requirements.txt
└── stadiff/
    ├── config.py         # constants, DEFAULT_ROI_BOX, run config
    ├── data.py           # folder -> train/val/test splits, grayscale/rectangular Dataset
    ├── masks.py          # fixed ROI rectangle (fixed_mask) + unused random masks
    ├── diffusion.py      # noise schedule, per-pixel timestep map, forward noising
    ├── model.py          # mask/timestep-map conditioned U-Net + EMA
    ├── sampler.py        # staged two-phase reverse process (per-pixel DDIM)
    ├── metrics.py        # PSNR (per region) and SSIM
    ├── evaluation.py     # validation/test loop
    ├── anomaly.py        # anomaly maps, image scores, pixel AUROC
    ├── visualize.py      # result grids + phase illustrations
    └── utils.py          # device, seeding, model building, checkpoint loading
```

## Outputs

```
runs/exp1/
├── splits.json       # which files are train / val / test (re-used on resume and by test.py)
├── best.pt           # EMA weights with the lowest validation eps-MSE
├── last.pt           # full state for --resume
├── val_log.jsonl     # all validation metrics over time
├── samples/          # reconstruction grids at every validation
└── test/             # written by test.py: metrics json, per-image csv, example grid
```

## Notes

- All images are preprocessed ONCE at start-up and kept in memory (cached as `runs/<exp>/cache_*.npz`), so slow storage
  such as Google Drive is only read the first time. Memory = N x H x W bytes (a few hundred MB for thousands of images).
  `--workers 0` (default) is fastest. If you change `--H/--W/--norm` or the images, the cache rebuilds automatically.
- The ROI rectangle must cover every weld in every image. The vertical position of the welds varies between images,
  so the rectangle is large; always check it with `preview_roi.py` on many images (increase `--n`).
- `best.pt` is selected by validation eps-MSE; check `val_log.jsonl` if PSNR disagrees.
- The "corrupted" row in the metrics is the no-model baseline; the model should beat it in both ROI and background.
- Higher resolution (e.g. `--H 384 --W 512`) keeps weld detail but needs much more memory and training time.
