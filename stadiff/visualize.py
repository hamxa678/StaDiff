"""Plotting helpers."""
import os

from .metrics import to01

COLUMNS = [("orig", "Original"), ("mask", "ROI mask"), ("corrupted", "Corrupted"),
           ("recon", "StaDiff recon"), ("composite", "Recon + clean BG")]


def save_grid(ex, path):
    """Save a rows-of-examples figure. `ex` maps column name -> tensor (N,C,H,W) in [-1,1] (mask in [0,1])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = ex["orig"].shape[0]
    fig, axes = plt.subplots(n, len(COLUMNS), figsize=(3 * len(COLUMNS), 3 * n), squeeze=False)
    for i in range(n):
        for j, (k, title) in enumerate(COLUMNS):
            ax = axes[i, j]
            if k == "mask":
                ax.imshow(ex[k][i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(to01(ex[k][i]).permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0:
                ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)
