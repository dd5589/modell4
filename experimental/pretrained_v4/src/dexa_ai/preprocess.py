from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

IMAGE_SIZE = 160

def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo: hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)

def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(normalize_uint8(arr), mode='L')

def letterbox(pil: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return ImageOps.pad(pil, (size, size), method=Image.Resampling.BILINEAR, color=0, centering=(0.5,0.5))

def tensor_from_array(arr: np.ndarray, train: bool=False, seed: int | None=None) -> torch.Tensor:
    pil = letterbox(to_pil(arr), IMAGE_SIZE)
    if train:
        # Light augmentation: do not vertically flip medical anatomy.
        if np.random.random() < 0.30:
            pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        angle = float(np.random.uniform(-4.0,4.0))
        if abs(angle) > 0.5:
            pil = pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        if np.random.random() < 0.15:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=0.6))
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).unsqueeze(0)
    return x
