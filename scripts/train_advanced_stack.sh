#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: $0 DATA_ROOT LABELS_XLSX OUTPUT_DIR}"
LABELS_XLSX="${2:?Usage: $0 DATA_ROOT LABELS_XLSX OUTPUT_DIR}"
OUTPUT_DIR="${3:-artifacts/lgbm_bag}"
TMP_FEATURES="${OUTPUT_DIR}/../lgbm_features.npy"
TMP_PATHS="${OUTPUT_DIR}/../lgbm_paths.npy"
mkdir -p "${OUTPUT_DIR}"
python scripts/make_lgbm_features.py \
  --data-root "${DATA_ROOT}" \
  --labels-xlsx "${LABELS_XLSX}" \
  --out-features "${TMP_FEATURES}" \
  --out-paths "${TMP_PATHS}"
python scripts/train_lgbm_bag.py \
  --data-root "${DATA_ROOT}" \
  --labels-xlsx "${LABELS_XLSX}" \
  --features "${TMP_FEATURES}" \
  --out-dir "${OUTPUT_DIR}"
rm -f "${TMP_FEATURES}" "${TMP_PATHS}"
