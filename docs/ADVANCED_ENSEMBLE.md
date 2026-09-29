# Advanced ensemble v0.3

The production output contract is unchanged. The internal predictor now combines:

1. three independently trained regional CNN members;
2. HOG classical members;
3. task-specific clinical/geometric experts;
4. a 3-fold + full LightGBM bagger trained on multi-scale grayscale, HOG, LBP, intensity-grid and shape/edge features.

The bagger contributes four members per target (three fold models plus one full-data model), i.e. 28 additional lightweight ensemble members across the seven target heads.

## Why this is stronger

The original deep ensemble is intentionally small and was trained from scratch on a dataset of roughly 100 studies. The new stack adds a different hypothesis class and different image representation instead of duplicating the same CNN. This is expected to reduce correlated errors while keeping inference local and fast.

## Evidence

The 3-fold bagger was evaluated at study level. Because rare targets have very small support, metrics are engineering evidence rather than clinical validation.

- spine quality: OOF F1 0.516, ROC-AUC 0.524, PR-AUC 0.357
- spine placement: OOF F1 0.571, ROC-AUC 0.801, PR-AUC 0.362
- spine axis: OOF F1 0.316, ROC-AUC 0.606, PR-AUC 0.194
- spine artifact: OOF F1 0.421, ROC-AUC 0.608, PR-AUC 0.252
- hip quality: OOF F1 0.600, ROC-AUC 0.644, PR-AUC 0.489
- hip rotation: OOF F1 0.528, ROC-AUC 0.655, PR-AUC 0.483
- hip ROI: OOF F1 0.286, ROC-AUC 0.510, PR-AUC 0.258

Do not report the engineering holdout 1.0 obtained by the combined full-data predictor as independent validation; full-data members touch the evaluation studies.

## Runtime output

The CSV/UI vocabulary remains exactly the organizer-confirmed contract:

- `anatomical_region`: `Поясничный отдел позвоночника` / `Проксимальный отдел бедра`
- spine violations: `Некорректная укладка`; `Не выравнена ось позвоночника`; `Присутствуют посторонние предметы`
- hip violations: `Некорректная укладка`; `Некорректная область интереса`
- `quality_prob` in [0,1]

The presentation and visible output design are unchanged.
