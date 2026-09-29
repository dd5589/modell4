# Advanced Ensemble Release Summary

Release family: DXA Quality AI
Internal version: 0.3 advanced stack

## Internal ensemble
- 6 regional CNN members (3 per anatomy)
- 7 HOG heads
- 4 task-specific clinical/geometry experts
- 28 LightGBM members (3 fold models + 1 full model x 7 target heads)
- 45 internal members total

## Public contract
Unchanged. The API and CSV retain the same field names and organizer-confirmed Russian vocabulary. `quality_prob` remains in [0,1].

## Evidence
LightGBM-only scores are provided as 3-fold study-level OOF engineering evidence in `reports/advanced_ensemble/lgbm_oof_all.json`. Because the dataset has only about 100 studies and several rare targets have <=10 positive studies, these are not clinical validation results.

A combined score calculated with full-data members on the visible split is deliberately not reported as an independent metric because it would leak training information into evaluation.

## Target host
The remaining acceptance gates are unchanged: Docker build on the target Linux/H200 host, CUDA/H200 benchmark, and organizer-side hidden closed test.
