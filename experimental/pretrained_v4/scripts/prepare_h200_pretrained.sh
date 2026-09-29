#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: $0 DATA_ROOT LABELS_XLSX OUT_DIR [EPOCHS]}"
LABELS="${2:?Usage: $0 DATA_ROOT LABELS_XLSX OUT_DIR [EPOCHS]}"
OUT="${3:-artifacts}"
EPOCHS="${4:-12}"
export PYTHONPATH="$(pwd)/src"
python scripts/train_pretrained_ensemble.py \
  --data-root "$DATA_ROOT" \
  --labels-xlsx "$LABELS" \
  --out-dir "$OUT" \
  --region both \
  --backbones resnet50,densenet121,efficientnet_b0,convnext_tiny \
  --epochs "$EPOCHS" \
  --freeze-epochs 2
