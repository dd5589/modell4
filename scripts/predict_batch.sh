#!/usr/bin/env bash
set -euo pipefail
python -m dexa_ai.batch --input "$1" --weights artifacts --out-csv results.csv --workers "${WORKERS:-1}"
