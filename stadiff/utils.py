"""Small helpers: device, seeding, checkpoint loading."""
import random

import numpy as np
import torch

from .model import UNet


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(ckpt_path, dev):
    """Load the EMA weights + config from a checkpoint saved by train.py (best.pt or last.pt)."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = UNet(base=cfg["base"]).to(dev)
    model.load_state_dict(ck["ema"])
    return model.eval(), cfg, ck
