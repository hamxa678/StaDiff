# StaDiff - Staged Diffusion with ROI-dependent noise horizons

The region of interest (ROI) is noised up to `t_roi` (e.g. 500), the background only up to `t_bg` (e.g. 300).
The reverse process is staged: Phase 1 denoises only the ROI until it reaches the background's level,
Phase 2 denoises the whole image together. The ROI can be anywhere in the image.

## Project layout

```
stadiff_project/
├── train.py              # entry point: train + validate, saves best.pt / last.pt
├── test.py               # entry point: evaluate on the held-out test split
├── infer.py              # entry point: run on one image with your own ROI box
├── make_demo_data.py     # entry point: toy dataset for trying the pipeline
├── requirements.txt
└── stadiff/              # the library
    ├── config.py         # constants (schedule length, image extensions) + run config
    ├── data.py           # folder -> train/val/test splits, Dataset, demo data
    ├── masks.py          # random ROI masks (training/eval) and box masks (inference)
    ├── diffusion.py      # noise schedule, per-pixel timestep map, forward noising
    ├── model.py          # mask/timestep-map conditioned U-Net + EMA
    ├── sampler.py        # staged two-phase reverse process (per-pixel DDIM)
    ├── metrics.py        # PSNR (per region) and SSIM
    ├── evaluation.py     # validation/test loop
    ├── visualize.py      # result grids
    └── utils.py          # device, seeding, checkpoint loading
```

## Quick start

```bash
pip install -r requirements.txt

# optional toy data
python make_demo_data.py --out demo_images --n 80

# train (use a GPU and --amp for real runs)
python train.py --data_dir path/to/images --out runs/exp1 --size 64 --t_roi 500 --t_bg 300 --steps 20000

# test on the held-out split
python test.py --ckpt runs/exp1/best.pt

# one image, your own ROI (fractions x0 y0 x1 y1)
python infer.py --ckpt runs/exp1/best.pt --image photo.jpg --box 0.4 0.4 0.8 0.8
```

## What happens in training (train.py)

1. Take a clean image and a random ROI mask (random position, size, shape).
2. Draw a global step `s ~ U[1, t_roi]`.
3. Per-pixel timestep: ROI pixels `t = s`, background pixels `t = min(s, t_bg)`.
4. Noise every pixel with its own `alpha_bar[t]`.
5. The network gets the noisy image, the mask and the timestep map, and predicts the noise.
6. MSE between predicted and true noise.

The staging (background untouched while `s > t_bg`) lives in `sampler.py`; training only has to cover every state the sampler visits.

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

- `--size` must be divisible by 4. Start with 64.
- `best.pt` is selected by validation eps-MSE (cheap and stable); check `val_log.jsonl` if PSNR disagrees.
- The "corrupted" row in the metrics is the no-model baseline; the model should beat it in both ROI and background.
- If your real ROIs do not look like random rectangles/ellipses, change `random_masks` in `stadiff/masks.py`.
# StaDiff
