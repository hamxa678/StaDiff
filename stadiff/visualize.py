"""Plotting helpers (grayscale or RGB, any aspect ratio)."""
import os

from .metrics import to01

COLUMNS = [("orig", "Original"), ("mask", "ROI mask"), ("corrupted", "Corrupted"),
           ("recon", "StaDiff recon"), ("composite", "Recon + clean BG")]


def _imshow(ax, x):
    """x: (C,H,W) tensor in [-1,1]."""
    a = to01(x).cpu()
    if a.shape[0] == 1:
        ax.imshow(a[0].numpy(), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(a.permute(1, 2, 0).numpy())


def save_grid(ex, path):
    """Save a rows-of-examples figure. `ex` maps column name -> tensor (N,C,H,W) in [-1,1] (mask in [0,1])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, _, H, W = ex["orig"].shape
    cw = 3.2
    fig, axes = plt.subplots(n, len(COLUMNS), figsize=(cw * len(COLUMNS), (cw * H / W + 0.35) * n), squeeze=False)
    for i in range(n):
        for j, (k, title) in enumerate(COLUMNS):
            ax = axes[i, j]
            if k == "mask":
                ax.imshow(ex[k][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                _imshow(ax, ex[k][i])
            ax.axis("off")
            if i == 0:
                ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase illustrations: show that the background is NOT changed during Phase 1
# ---------------------------------------------------------------------------
def _pick(snaps, s_target):
    return min(snaps, key=lambda it: abs(it[0] - s_target))


def save_filmstrip(snaps, x_start, m, cfg, targets, path, diff_vmax=0.5):
    """
    Rows: (1) the image at step s with the ROI outline, (2) |x_s - corrupted start| (where did pixels change?),
    (3) the per-pixel noise level t. Columns: chosen steps s. Phase 1 columns have orange titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from .diffusion import build_t_map

    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    H, W = x_start.shape[-2:]
    chosen = [_pick(snaps, t) for t in targets]
    fig, axes = plt.subplots(3, len(chosen), figsize=(2.6 * len(chosen), 3 * (2.6 * H / W) + 1.8), squeeze=False)
    mk = m[0, 0].cpu().numpy()
    for j, (s, x) in enumerate(chosen):
        diff = (x - x_start)[0].abs().mean(0).cpu().numpy()
        t_map = build_t_map(torch.full((1,), int(s)), m.cpu(), t_roi, t_bg)[0, 0].numpy()
        _imshow(axes[0, j], x[0])
        axes[0, j].contour(mk, levels=[0.5], colors="lime", linewidths=1.2)
        axes[1, j].imshow(diff, cmap="inferno", vmin=0, vmax=diff_vmax)
        im = axes[2, j].imshow(t_map, cmap="viridis", vmin=0, vmax=t_roi)
        phase1 = s > t_bg
        axes[0, j].set_title(f"s = {s}\n{'Phase 1: ROI only' if phase1 else 'Phase 2: whole image'}",
                             fontsize=10, color="darkorange" if phase1 else "tab:blue", fontweight="bold")
        for i in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("image\n(green = ROI outline)", fontsize=10)
    axes[1, 0].set_ylabel("change vs corrupted start\n(black = untouched)", fontsize=10)
    axes[2, 0].set_ylabel("noise level t per pixel", fontsize=10)
    fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01, label="t")
    fig.suptitle("Reverse process: in Phase 1 only the ROI changes; the background stays exactly as it was", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_phase_curves(s_list, curves, cfg, path):
    """curves: {'level'|'change'|'psnr': {region_name: [value per s]}}. x-axis runs from s=t_roi to s=0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_roi, t_bg = cfg["t_roi"], cfg["t_bg"]
    titles = {"level": ("Noise level t of each region", "mean timestep t"),
              "change": ("Change vs corrupted start", "mean |x_s - x_start|"),
              "psnr": ("Closeness to the clean original", "PSNR vs original (dB)")}
    colors = {0: "tab:red", 1: "tab:orange", 2: "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, key in zip(axes, ["level", "change", "psnr"]):
        for k, (name, vals) in enumerate(curves[key].items()):
            ax.plot(s_list, vals, marker="o", ms=3, lw=2, label=name, color=colors[k])
        ax.axvspan(t_bg, t_roi, color="orange", alpha=0.10)
        ax.axvspan(0, t_bg, color="tab:blue", alpha=0.08)
        ax.axvline(t_bg, color="k", ls="--", lw=1)
        ax.set_title(titles[key][0]); ax.set_ylabel(titles[key][1]); ax.set_xlabel("global step s  (time runs left -> right)")
        ax.invert_xaxis(); ax.grid(alpha=0.3)
        ymax = ax.get_ylim()[1]
        ax.text((t_roi + t_bg) / 2, ymax, "Phase 1", ha="center", va="top", fontsize=10, color="darkorange")
        ax.text(t_bg / 2, ymax, "Phase 2", ha="center", va="top", fontsize=10, color="tab:blue")
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_gif(snaps, x_start, m, cfg, path, size=320, ms=250):
    """Animated GIF (good for slides): left = image with ROI outline, right = change vs corrupted start."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw

    cmap, t_bg = plt.get_cmap("inferno"), cfg["t_bg"]
    H, W = x_start.shape[-2:]
    Wo, Ho = size, int(round(size * H / W))
    mk = np.array(Image.fromarray((m[0, 0].cpu().numpy() * 255).astype(np.uint8)).resize((Wo, Ho), Image.BILINEAR)) > 127
    edge = mk & ~(np.roll(mk, 1, 0) & np.roll(mk, -1, 0) & np.roll(mk, 1, 1) & np.roll(mk, -1, 1))
    frames = []
    for s, x in snaps:
        a = to01(x[0]).cpu().numpy()                                 # (C,H,W)
        img = (a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)) * 255
        img = np.array(Image.fromarray(img.astype(np.uint8)).convert("RGB").resize((Wo, Ho), Image.BICUBIC))
        img[edge] = (0, 255, 0)
        diff = (x - x_start)[0].abs().mean(0).cpu().numpy() / 0.5
        heat = (cmap(np.clip(diff, 0, 1))[..., :3] * 255).astype(np.uint8)
        heat = np.array(Image.fromarray(heat).resize((Wo, Ho), Image.NEAREST))
        canvas = Image.new("RGB", (2 * Wo, Ho + 28), (255, 255, 255))
        canvas.paste(Image.fromarray(img), (0, 28))
        canvas.paste(Image.fromarray(heat), (Wo, 28))
        phase = "Phase 1: ROI only" if s > t_bg else "Phase 2: whole image"
        ImageDraw.Draw(canvas).text((4, 8), f"s = {s:3d}   {phase}      right panel: change vs corrupted start", fill=(0, 0, 0))
        frames.append(canvas)
    frames += [frames[-1]] * 4
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


# ---------------------------------------------------------------------------
# Anomaly-testing figures
# ---------------------------------------------------------------------------
def save_anomaly_examples(rows, mask, path, vmax):
    """
    rows: list of dicts {x0 (C,H,W), rec (C,H,W), amap (1,H,W), title, gt (H,W) bool or None}, values in [-1,1].
    Columns: original (red outline = ground-truth defect, if given) | reconstruction | anomaly map (green = ROI).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    H, W = rows[0]["x0"].shape[-2:]
    cw = 4.0
    fig, axes = plt.subplots(n, 3, figsize=(cw * 3, (cw * H / W + 0.45) * n), squeeze=False)
    mk = mask[0, 0].cpu().numpy()
    for i, r in enumerate(rows):
        _imshow(axes[i, 0], r["x0"])
        if r.get("gt") is not None and r["gt"].any():
            axes[i, 0].contour(r["gt"].numpy().astype(float), levels=[0.5], colors="red", linewidths=1.2)
        _imshow(axes[i, 1], r["rec"])
        axes[i, 2].imshow(r["amap"][0].float().numpy(), cmap="inferno", vmin=0, vmax=vmax)
        axes[i, 2].contour(mk, levels=[0.5], colors="lime", linewidths=1)
        axes[i, 0].set_title(r["title"], fontsize=10, loc="left")
        axes[i, 1].set_title("StaDiff reconstruction", fontsize=10)
        axes[i, 2].set_title("anomaly map |input - recon|", fontsize=10)
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)
