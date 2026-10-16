"""
Evaluate a trained model on the held-out TEST split (never seen during training or model selection).

  python test.py --ckpt runs/exp1/best.pt
Outputs (in runs/exp1/test/): test_metrics.json, test_per_image.csv, test_examples.png
"""
import argparse
import csv
import json
from pathlib import Path

from torch.utils.data import DataLoader

from stadiff.data import ImgDS
from stadiff.diffusion import make_ab
from stadiff.evaluation import evaluate, print_summary
from stadiff.utils import get_device, load_model
from stadiff.visualize import save_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", default=None, help="splits.json (default: next to the checkpoint)")
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=50, help="number of reverse steps")
    p.add_argument("--n_examples", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    dev = get_device()
    model, cfg, ck = load_model(a.ckpt, dev)
    split_path = Path(a.splits) if a.splits else Path(a.ckpt).parent / "splits.json"
    files = json.loads(split_path.read_text())["test"]
    out = Path(a.out) if a.out else Path(a.ckpt).parent / "test"
    out.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(ImgDS(files, cfg), batch_size=min(a.batch, len(files)), shuffle=False)
    print(f"testing {a.ckpt} (trained {ck['step']} steps) on {len(files)} held-out images, cfg={cfg}")

    sm, rows, ex = evaluate(model, dl, cfg, make_ab(dev), dev, a.sample_steps, seed=a.seed,
                            n_examples=a.n_examples, collect_rows=True)
    print_summary(sm, ind="  ")

    (out / "test_metrics.json").write_text(json.dumps({"cfg": cfg, "n_images": len(files), **sm}, indent=1))
    with open(out / "test_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if ex:
        save_grid(ex, out / "test_examples.png")
    print(f"saved metrics, per-image CSV and example grid to {out}/")


if __name__ == "__main__":
    main()
