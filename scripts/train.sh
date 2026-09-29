#!/usr/bin/env bash
set -euo pipefail
python -m dexa_ai.train_region --data-root "$1" --out-dir artifacts --region spine --epochs "${EPOCHS:-10}"
python -m dexa_ai.train_region --data-root "$1" --out-dir artifacts --region hip --epochs "${EPOCHS:-10}"
python -m dexa_ai.train_hog --data-root "$1" --out-dir artifacts --region spine
python -m dexa_ai.train_hog --data-root "$1" --out-dir artifacts --region hip
