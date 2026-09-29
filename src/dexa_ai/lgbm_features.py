from __future__ import annotations
import cv2
import numpy as np
from skimage import feature, filters

PIXEL_SIZE_X_MM = 0.60
PIXEL_SIZE_Y_MM = 1.05


def normalize_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    vals = a[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def small_image_features(arr: np.ndarray) -> np.ndarray:
    """Exact compact feature recipe used to train the LightGBM bagger."""
    x = normalize_array(arr)
    x = cv2.resize(x, (160, 192), interpolation=cv2.INTER_AREA)
    parts: list[np.ndarray] = [
        cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA).ravel(),
        cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA).ravel(),
        feature.hog(
            (x * 255).astype(np.uint8),
            orientations=6,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        ),
    ]
    lbp = feature.local_binary_pattern((x * 255).astype(np.uint8), 8, 2, method="uniform")
    parts.append(np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)[0])

    stats: list[float] = []
    for gy, gx in ((4, 4), (8, 8)):
        for iy in range(gy):
            for ix in range(gx):
                q = x[iy * 192 // gy : (iy + 1) * 192 // gy,
                      ix * 160 // gx : (ix + 1) * 160 // gx]
                stats += [float(q.mean()), float(q.std()), float(np.percentile(q, 90)), float((q > 0.8).mean())]
    parts.append(np.asarray(stats, dtype=np.float32))

    sob = filters.sobel(x)
    parts.append(np.asarray([
        float(sob.mean()), float(sob.std()), float(np.percentile(sob, 90)),
        float(np.mean(np.abs(x - np.fliplr(x)))),
    ], dtype=np.float32))

    for threshold in (0.6, 0.8, 0.9):
        mask = x > threshold
        if mask.any():
            yy, xx = np.nonzero(mask)
            parts.append(np.asarray([
                float(mask.mean()), float(xx.mean() / 160.0), float(yy.mean() / 192.0),
                float((xx.max() - xx.min() + 1) / 160.0),
                float((yy.max() - yy.min() + 1) / 192.0),
            ], dtype=np.float32))
        else:
            parts.append(np.zeros(5, dtype=np.float32))
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts]).astype(np.float32)
