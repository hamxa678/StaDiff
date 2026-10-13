"""Global constants and run configuration."""

T_SCHED = 1000                                    # length of the underlying DDPM noise schedule
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_cfg(size, base, t_roi, t_bg):
    """Configuration that is stored inside every checkpoint (so test/infer need no extra flags)."""
    assert 0 < t_bg < t_roi <= T_SCHED, "need 0 < t_bg < t_roi <= 1000"
    assert size % 4 == 0, "image size must be divisible by 4"
    return {"size": size, "base": base, "t_roi": t_roi, "t_bg": t_bg}
