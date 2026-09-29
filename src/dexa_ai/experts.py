"""Task-specific expert heads for rare denstometry QC labels.

The project intentionally keeps these experts interpretable:
- spine axis: deterministic centerline geometry tied to the 5° criterion;
- spine placement: coverage/field-of-view classifier;
- spine artifact: HOG + geometry tree ensemble;
- hip rotation: HOG + geometry logistic head;
- hip ROI: border/coverage classifier.

All learned expert heads are trained at study level (image samples are weighted
so one study contributes approximately one total unit of sample weight).
"""
from __future__ import annotations
import math, pickle, json, os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage import transform, feature, filters, morphology
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .dicomio import read_dicom
from .labels import build_manifest

# Organizer-confirmed acquisition geometry: the dataset uses one scanner and
# DICOM may not contain PixelSpacing. These constants are therefore part of
# the frozen inference contract, not values read from patient metadata.
PIXEL_SIZE_X_MM = 0.60
PIXEL_SIZE_Y_MM = 1.05


def physical_margin_pixels(px_x_mm: float = PIXEL_SIZE_X_MM, px_y_mm: float = PIXEL_SIZE_Y_MM) -> tuple[int, int]:
    """Return horizontal/vertical pixel margins for the organizer's 2 cm/3 cm rule."""
    x = max(1, int(round(20.0 / px_x_mm)))  # 2 cm
    y = max(1, int(round(30.0 / px_y_mm)))  # 3 cm
    return x, y


def normalize_array(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    vals = a[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def resized(arr: np.ndarray) -> np.ndarray:
    return transform.resize(normalize_array(arr), (192, 160), preserve_range=True, anti_aliasing=True).astype(np.float32)


def centerline_angle_deg(arr: np.ndarray, pixel_size_x_mm: float = PIXEL_SIZE_X_MM, pixel_size_y_mm: float = PIXEL_SIZE_Y_MM) -> float:
    """Estimate the spine centerline tilt in physical coordinates.

    The angle is measured relative to the vertical physical axis, accounting
    for the scanner's anisotropic pixel size.
    """
    x = normalize_array(arr)
    h, w = x.shape
    left, right = int(.15 * w), int(.85 * w)
    x = x[:, left:right]
    centers, ys = [], []
    # Use a high-intensity ridge proxy. Robust against black background.
    for y in range(0, h, max(1, h // 140)):
        row = x[y]
        thr = np.quantile(row, .80)
        idx = np.where(row >= thr)[0]
        if idx.size:
            weights = np.square(row[idx])
            centers.append(float(np.sum(idx * weights) / (np.sum(weights) + 1e-8)))
            ys.append(y)
    if len(centers) < 12:
        return 0.0
    yy = np.asarray(ys, dtype=np.float32)
    cc = np.asarray(centers, dtype=np.float32)
    lo, hi = int(.10 * len(yy)), int(.90 * len(yy))
    yy, cc = yy[lo:hi], cc[lo:hi]
    if len(yy) < 8:
        return 0.0
    slope_px = float(np.polyfit(yy, cc, 1)[0])
    # x/y pixel slope -> physical x/y slope.
    slope_phys = slope_px * float(pixel_size_x_mm) / max(float(pixel_size_y_mm), 1e-8)
    return float(math.degrees(math.atan(abs(slope_phys))))


def axis_probability(arr: np.ndarray, scale_deg: float = 1.5) -> float:
    angle = centerline_angle_deg(arr)
    return float(1.0 / (1.0 + math.exp(-(angle - 5.0) / scale_deg)))


def placement_features(arr: np.ndarray) -> np.ndarray:
    x = resized(arr)
    h, w = x.shape
    central = x[:, int(.15*w):int(.85*w)]
    feats: list[float] = []
    for frac in (.10, .15, .20, .25):
        k = max(1, int(h*frac))
        top = central[:k]
        bottom = central[-k:]
        feats += [float(top.mean()), float(bottom.mean()),
                  float((top > .80).mean()), float((bottom > .80).mean()),
                  float(abs(top.mean()-bottom.mean()))]
    rowmean = central.mean(axis=1)
    feats += [float(np.mean(np.percentile(rowmean, [50, 75, 90]))),
              float(rowmean[:max(1, int(.15*h))].max()),
              float(rowmean[-max(1, int(.15*h)):].max())]
    return np.asarray(feats, dtype=np.float32)


def roi_features(arr: np.ndarray) -> np.ndarray:
    x = resized(arr)
    h, w = x.shape
    m = x > .80
    feats: list[float] = []
    # Explicit physical-margin features derived from the organizer-confirmed
    # scanner scale. These are FOV/coverage proxies: without an explicit ROI
    # contour in the input, they cannot claim pixel-perfect ROI localization.
    margin_x, margin_y = physical_margin_pixels()
    margin_x = min(margin_x, max(1, w // 2))
    margin_y = min(margin_y, max(1, h // 2))
    top3, bottom3 = m[:margin_y], m[-margin_y:]
    left2, right2 = m[:, :margin_x], m[:, -margin_x:]
    feats += [float(top3.mean()), float(bottom3.mean()), float(left2.mean()), float(right2.mean()),
              float(1.0 - top3.mean()), float(1.0 - bottom3.mean()),
              float(1.0 - left2.mean()), float(1.0 - right2.mean())]
    for frac in (.05, .10, .15, .20):
        hh, ww = max(1, int(h*frac)), max(1, int(w*frac))
        top, bottom = m[:hh], m[-hh:]
        left, right = m[:, :ww], m[:, -ww:]
        feats += [float(top.mean()), float(bottom.mean()), float(left.mean()), float(right.mean()),
                  float(top.mean()+bottom.mean()+left.mean()+right.mean())]
    if m.any():
        yy, xx = np.nonzero(m)
        feats += [float(yy.min()/h), float((h-1-yy.max())/h), float(xx.min()/w), float((w-1-xx.max())/w)]
    else:
        feats += [1.0, 1.0, 1.0, 1.0]
    return np.asarray(feats, dtype=np.float32)


def vision_features(arr: np.ndarray) -> np.ndarray:
    x = resized(arr)
    h = feature.hog(x, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
    extra: list[float] = []
    for q in (60, 70, 80, 90):
        m = x > np.percentile(x, q)
        m = morphology.remove_small_objects(m, max_size=20)
        if m.any():
            yy, xx = np.nonzero(m)
            cx, cy = xx.mean()/160.0, yy.mean()/192.0
            ww, hh = (xx.max()-xx.min()+1)/160.0, (yy.max()-yy.min()+1)/192.0
            xy = np.c_[xx, yy] - np.c_[xx, yy].mean(axis=0)
            cov = np.cov(xy, rowvar=False) if len(xy) > 2 else np.eye(2)
            vals, vecs = np.linalg.eigh(cov)
            v = vecs[:, int(np.argmax(vals))]
            angle = math.degrees(math.atan2(float(v[1]), float(v[0])))
            dev = abs(90.0 - abs(angle))
            extra += [float(m.mean()), float(cx), float(cy), float(ww), float(hh), float(dev)]
        else:
            extra += [0.0] * 6
    sob = filters.sobel(x)
    extra += [float(sob.mean()), float(np.percentile(sob, 90)),
              float(np.abs(np.diff(x, axis=1)).mean()), float(np.abs(np.diff(x, axis=0)).mean())]
    return np.concatenate([h.astype(np.float32), np.asarray(extra, dtype=np.float32)])


def _study_label(row: dict, target: str) -> int | None:
    if target == 'spine_placement': return row.get('spine_placement')
    if target == 'spine_axis': return row.get('spine_axis')
    if target == 'spine_artifact': return row.get('spine_artifact')
    if target == 'hip_rotation':
        vals = [row.get('right_rotation'), row.get('left_rotation')]
        vals = [int(v) for v in vals if v in (0,1)]
        return max(vals) if vals else None
    if target == 'hip_roi':
        vals = [row.get('right_roi'), row.get('left_roi')]
        vals = [int(v) for v in vals if v in (0,1)]
        return max(vals) if vals else None
    return None


def _fit_weighted(model, X, y, study_ids):
    counts = defaultdict(int)
    for s in study_ids: counts[s] += 1
    weights = np.asarray([1.0 / counts[s] for s in study_ids], dtype=np.float32)
    if hasattr(model, 'named_steps'):
        final_name = list(model.named_steps.keys())[-1]
        model.fit(X, y, **{f'{final_name}__sample_weight': weights})
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def _make_model(target: str):
    if target in ('spine_placement', 'hip_roi', 'hip_rotation'):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight='balanced', C=0.7))
    if target == 'spine_artifact':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    raise ValueError(target)


def fit_experts(data_root: str, labels_xlsx: str | None, out_dir: str) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(data_root, labels_xlsx=labels_xlsx)
    cache: dict[str, np.ndarray] = {}
    for r in rows:
        p = r['path']
        if p not in cache:
            cache[p] = normalize_array(read_dicom(p).array)

    configs = {
        'spine_placement': placement_features,
        'spine_artifact': vision_features,
        'hip_rotation': vision_features,
        'hip_roi': roi_features,
    }
    metrics = {}
    for target, extractor in configs.items():
        region = 'spine' if target.startswith('spine_') else 'hip'
        rrows = [r for r in rows if r['region'] == region and _study_label(r, target) is not None]
        by_study = defaultdict(list)
        for r in rrows: by_study[r['study']].append(r)
        X, y, sids = [], [], []
        for study, items in by_study.items():
            label = _study_label(items[0], target)
            if label is None: continue
            for r in items:
                X.append(extractor(cache[r['path']]))
                y.append(int(label)); sids.append(study)
        X, y = np.stack(X), np.asarray(y, dtype=np.int64)
        model = _make_model(target)
        _fit_weighted(model, X, y, sids)
        with open(out / f'{target}.pkl', 'wb') as f: pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        metrics[target] = {'n_images': int(len(y)), 'n_studies': int(len(by_study)), 'positive_studies': int(sum(_study_label(v[0], target) for v in by_study.values()))}

    # The axis expert is deterministic by design, so save only its calibration.
    cfg = {
        'version': '0.2-expert',
        'axis_rule': {'criterion_deg': 5.0, 'sigmoid_scale_deg': 1.5},
        'target_weights': {
            'spine_placement': {'deep': 0.35, 'expert': 0.65},
            'spine_axis': {'deep': 0.20, 'expert': 0.80},
            'spine_artifact': {'deep': 0.35, 'expert': 0.65},
            'hip_rotation': {'deep': 0.30, 'expert': 0.70},
            'hip_roi': {'deep': 0.25, 'expert': 0.75},
        },
        'metrics_inventory': metrics,
    }
    json.dump(cfg, open(out/'expert_config.json','w',encoding='utf8'), ensure_ascii=False, indent=2)
    return cfg


def expert_probability(target: str, arr: np.ndarray, expert_models: dict, cfg: dict) -> float:
    if target == 'spine_axis':
        return axis_probability(arr, float(cfg.get('axis_rule',{}).get('sigmoid_scale_deg',1.5)))
    if target not in expert_models:
        return 0.5
    if target in ('spine_placement','hip_roi'):
        feat = placement_features(arr) if target == 'spine_placement' else roi_features(arr)
    else:
        feat = vision_features(arr)
    return float(expert_models[target].predict_proba(np.asarray(feat, dtype=np.float32).reshape(1,-1))[:,1][0])
