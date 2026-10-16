"""Global constants and run configuration."""

T_SCHED = 1000                                    # length of the underlying DDPM noise schedule
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Hard-coded ROI rectangle as FRACTIONS of the image: (x0, y0, x1, y1), origin = top-left.
# Meant to cover all three welds and part of the rods. Check/tune it with:  python preview_roi.py --data_dir ...
DEFAULT_ROI_BOX = (0.15, 0.18, 0.87, 0.94)


def make_cfg(H, W, channels, base, t_roi, t_bg, roi_box, norm):
    """Configuration that is stored inside every checkpoint (so test/infer need no extra flags)."""
    assert 0 < t_bg < t_roi <= T_SCHED, "need 0 < t_bg < t_roi <= 1000"
    assert H % 4 == 0 and W % 4 == 0, "H and W must be divisible by 4"
    assert channels in (1, 3), "channels must be 1 (grayscale) or 3 (RGB)"
    x0, y0, x1, y1 = roi_box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "roi_box must be fractions with x0<x1, y0<y1"
    assert norm in ("clahe", "none")
    return {"H": H, "W": W, "channels": channels, "base": base, "t_roi": t_roi, "t_bg": t_bg,
            "roi_box": [float(v) for v in roi_box], "norm": norm}
