#!/usr/bin/env bash
set -euo pipefail
python -m dexa_ai.train_experts --data-root "$1" --labels-xlsx "${2:-$1/разметка.xlsx}" --out-dir artifacts/experts
