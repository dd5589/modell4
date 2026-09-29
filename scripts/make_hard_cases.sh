#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m dexa_ai.make_hard_cases --data-root "$1" --weights "${2:-artifacts}" --out-dir "${3:-reports/hard_cases}"
