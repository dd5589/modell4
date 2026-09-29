#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: $0 DATA_ROOT LABELS_XLSX OUTPUT_DIR [EPOCHS]}"
LABELS="${2:?Usage: $0 DATA_ROOT LABELS_XLSX OUTPUT_DIR [EPOCHS]}"
OUT="${3:-artifacts}"
EPOCHS="${4:-12}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is required for v0.4 H200 training'
print('CUDA:', torch.cuda.get_device_name(0))
print('VRAM_GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))
PY

python scripts/train_pretrained_ensemble.py \
  --data-root "$DATA_ROOT" \
  --labels-xlsx "$LABELS" \
  --out-dir "$OUT" \
  --region both \
  --backbones resnet50,densenet121,efficientnet_b0,convnext_tiny \
  --epochs "$EPOCHS" \
  --freeze-epochs 2

cat > "$OUT/pretrained_ensemble.json" <<JSON
{
  "version": "0.4-pretrained-ensemble",
  "enabled": false,
  "backbones": ["resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"],
  "target_weights": {
    "spine_quality": 0.0,
    "spine_placement": 0.0,
    "spine_axis": 0.0,
    "spine_artifact": 0.0,
    "hip_quality": 0.0,
    "hip_rotation": 0.0,
    "hip_roi": 0.0
  },
  "notes": "Run evaluate_v4_pretrained.py first. Enable only after same-holdout validation and manual QA."
}
JSON

echo 'v0.4 TRAINING COMPLETE. Do not enable pretrained blend yet.'
