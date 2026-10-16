"""
Train StaDiff on YOUR clean images with ONE fixed, hard-coded ROI rectangle (no random masks).

  python train.py --data_dir path/to/clean_images --out runs/exp1 --H 192 --W 256 --t_roi 500 --t_bg 300
  python train.py ... --roi_box 0.15 0.18 0.87 0.94     # ROI rectangle = fractions x0 y0 x1 y1 of the image
  python train.py ... --resume                          # continue an interrupted run

Look at the rectangle on your images first:  python preview_roi.py --data_dir path/to/clean_images
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stadiff.config import DEFAULT_ROI_BOX, make_cfg
from stadiff.data import ImgDS, make_splits
from stadiff.diffusion import build_t_map, make_ab, net_input, q_sample
from stadiff.evaluation import evaluate, print_summary
from stadiff.masks import fixed_mask
from stadiff.model import EMA
from stadiff.utils import build_model, get_device, set_seed
from stadiff.visualize import save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with your clean (anomaly-free) images")
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--size", type=int, default=None, help=argparse.SUPPRESS)   # removed option, see below
    p.add_argument("--H", type=int, default=192, help="image height after resizing (divisible by 4)")
    p.add_argument("--W", type=int, default=256, help="image width after resizing (divisible by 4)")
    p.add_argument("--channels", type=int, default=1, choices=[1, 3], help="1 = grayscale, 3 = RGB")
    p.add_argument("--norm", default="clahe", choices=["clahe", "none"], help="per-image exposure normalisation")
    p.add_argument("--roi_box", type=float, nargs=4, default=list(DEFAULT_ROI_BOX), metavar=("X0", "Y0", "X1", "Y1"),
                   help="fixed ROI rectangle as fractions of the image")
    p.add_argument("--base", type=int, default=64, help="U-Net base channels")
    p.add_argument("--t_roi", type=int, default=500, help="max noise level inside the ROI")
    p.add_argument("--t_bg", type=int, default=300, help="max noise level in the background")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is fastest: data is in memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--val_sampling_steps", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.size is not None:
        raise SystemExit("--size was replaced by --H and --W (images are rectangular now), "
                         "e.g. --H 192 --W 256  (or --H 96 --W 128 for a faster first run)")
    return a


def build_loaders(a, cfg, out):
    split_file = out / "splits.json"
    data_dir = os.path.abspath(a.data_dir)
    if split_file.exists():
        splits = json.loads(split_file.read_text())
        if splits.get("data_dir") != data_dir:
            raise SystemExit(f"{split_file} belongs to another dataset ({splits.get('data_dir')!r}).\n"
                             f"Use a new --out folder, or delete {out} to start fresh.")
        print(f"re-using splits from {split_file}")
    else:
        splits = make_splits(a.data_dir, a.val_frac, a.test_frac, a.seed)
        splits["data_dir"] = data_dir
        split_file.write_text(json.dumps(splits, indent=1))
    print(f"images  train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")
    bs = min(a.batch, len(splits["train"]))
    train_dl = DataLoader(ImgDS(splits["train"], cfg, out / "cache_train.npz", "train"), batch_size=bs, shuffle=True, drop_last=True,
                          num_workers=a.workers, persistent_workers=a.workers > 0)
    val_dl = DataLoader(ImgDS(splits["val"], cfg, out / "cache_val.npz", "val"), batch_size=min(bs, len(splits["val"])), shuffle=False)
    return train_dl, val_dl


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    a = parse_args()
    dev = get_device()
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a.H, a.W, a.channels, a.base, a.t_roi, a.t_bg, a.roi_box, a.norm)

    train_dl, val_dl = build_loaders(a, cfg, out)
    m1 = fixed_mask(cfg, dev)                                    # the one hard-coded ROI mask (1,1,H,W)

    model = build_model(cfg).to(dev).train()
    ema = EMA(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    lr_fn = lambda st: (min(1.0, (st + 1) / a.warmup) if st < a.warmup else
                        0.1 + 0.45 * (1 + math.cos(math.pi * (st - a.warmup) / max(1, a.steps - a.warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    ab = make_ab(dev)

    step, best = 0, float("inf")
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.m.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from step {step}")
    use_amp = a.amp and dev == "cuda"
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  cfg={cfg}")
    print(f"ROI rectangle covers {(m1 > 0.5).float().mean().item() * 100:.0f}% of the image")

    log = open(out / "val_log.jsonl", "a")
    print("starting training (progress is printed every --log_every steps)...", flush=True)
    start_step, t0 = step, time.time()
    done = step >= a.steps
    while not done:
        for x0 in train_dl:
            step += 1
            x0 = x0.to(dev, non_blocking=True)
            B = x0.shape[0]

            # --- one StaDiff training step (fixed ROI rectangle) ---------------------------------
            m = m1.expand(B, -1, -1, -1)
            s = torch.randint(1, a.t_roi + 1, (B,), device=dev)  # global step ~ U[1, t_roi]
            t_map = build_t_map(s, m, a.t_roi, a.t_bg)           # ROI: s,  background: min(s, t_bg)
            x_t, eps = q_sample(x0, t_map, ab)                   # per-pixel forward noising
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(net_input(x_t, m, t_map, a.t_roi), s)
            loss = F.mse_loss(pred.float(), eps)                 # predict the noise at every pixel
            # --------------------------------------------------------------------------------------

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model, step)

            if step % a.log_every == 0 or step == start_step + 1:
                speed = (step - start_step) / (time.time() - t0)
                print(f"step {step:7d}/{a.steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  {speed:.1f} it/s")
            if step % a.val_every == 0 or step >= a.steps:
                print(f"  [val @ step {step}] running validation...", flush=True)
                sm, _, ex = evaluate(ema.m, val_dl, cfg, ab, dev, a.val_sampling_steps,
                                     max_batches=a.val_batches, n_examples=4)
                print_summary(sm)
                log.write(json.dumps({"step": step, **sm}) + "\n"); log.flush()
                save_grid(ex, out / "samples" / f"step_{step:07d}.png")
                if sm["val_eps_mse"] < best:
                    best = sm["val_eps_mse"]
                    torch.save({"cfg": cfg, "ema": ema.m.state_dict(), "step": step, "val_eps_mse": best},
                               out / "best.pt")
                    print(f"  new best (val eps-MSE {best:.4f}) -> {out / 'best.pt'}")
                torch.save({"cfg": cfg, "model": model.state_dict(), "ema": ema.m.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best},
                           out / "last.pt")
            if step >= a.steps:
                done = True
                break
    print("training finished.")


if __name__ == "__main__":
    main()
