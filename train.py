"""
Train StaDiff.

  python train.py --data_dir path/to/images --out runs/exp1 --size 64 --t_roi 500 --t_bg 300 --amp
  python train.py --data_dir path/to/images --out runs/exp1 --resume          # continue an interrupted run
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stadiff.config import make_cfg
from stadiff.data import ImgDS, make_splits
from stadiff.diffusion import build_t_map, make_ab, net_input, q_sample
from stadiff.evaluation import evaluate, print_summary
from stadiff.masks import random_masks, box_mask, gaussian_blur
from stadiff.model import EMA, UNet
from stadiff.utils import get_device, set_seed
from stadiff.visualize import save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default="runs/exp1")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--base", type=int, default=64, help="U-Net base channels")
    p.add_argument("--t_roi", type=int, default=500, help="max noise level inside the ROI")
    p.add_argument("--t_bg", type=int, default=250, help="max noise level in the background (e.g. 300)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--val_sampling_steps", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_loaders(a, out):
    split_file = out / "splits.json"
    if split_file.exists():
        splits = json.loads(split_file.read_text())
        print(f"re-using splits from {split_file}")
    else:
        splits = make_splits(a.data_dir, a.val_frac, a.test_frac, a.seed)
        split_file.write_text(json.dumps(splits, indent=1))
    print(f"images  train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")
    bs = min(a.batch, len(splits["train"]))
    train_dl = DataLoader(ImgDS(splits["train"], a.size, True), batch_size=bs, shuffle=True, drop_last=True,
                          num_workers=a.workers, persistent_workers=a.workers > 0)
    val_dl = DataLoader(ImgDS(splits["val"], a.size, False), batch_size=min(bs, len(splits["val"])), shuffle=False)
    return train_dl, val_dl


def main():
    a = parse_args()
    dev = get_device()
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a.size, a.base, a.t_roi, a.t_bg)

    train_dl, val_dl = build_loaders(a, out)

    model = UNet(base=a.base).to(dev).train()
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

    log = open(out / "val_log.jsonl", "a")
    done = step >= a.steps
    while not done:
        for x0 in train_dl:
            step += 1
            x0 = x0.to(dev, non_blocking=True)
            B, _, H, W = x0.shape

            # --- one StaDiff training step -------------------------------------------------
            # m = random_masks(B, H, W, dev)                       # random ROI, any position / size / shape
            m = box_mask((0.25, 0.25, 0.75, 0.75), H, W, dev)  # optional: user-defined ROI (centered box)
            u = torch.rand(B, 1, 1, 1, device=dev)               # 20% uniform-noise samples keep Phase 2 strong
            m = torch.where(u < 0.1, torch.zeros_like(m), torch.where(u < 0.2, torch.ones_like(m), m))
            s = torch.randint(1, a.t_roi + 1, (B,), device=dev)  # global step ~ U[1, t_roi]
            t_map = build_t_map(s, m, a.t_roi, a.t_bg)           # ROI: s,  background: min(s, t_bg)
            x_t, eps = q_sample(x0, t_map, ab)                   # per-pixel forward noising
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(net_input(x_t, m, t_map, a.t_roi), s)
            loss = F.mse_loss(pred.float(), eps)                 # predict the noise at every pixel
            # --------------------------------------------------------------------------------

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model, step)

            if step % a.log_every == 0:
                print(f"step {step:7d}/{a.steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
            if step % a.val_every == 0 or step >= a.steps:
                sm, _, ex = evaluate(ema.m, val_dl, cfg, ab, dev, a.val_sampling_steps,
                                     max_batches=a.val_batches, n_examples=4)
                print(f"  [val @ step {step}] (EMA weights)")
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
