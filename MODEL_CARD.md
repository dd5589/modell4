# DXA Quality AI — Model Card

## Intended use
Локальный помощник контроля качества DICOM DXA исследований. Он предназначен для автоматического выявления нарушений качества, описанных в конкурсном ТЗ, и не является диагностической системой или заменой экспертной оценки.

## Inputs
- DICOM, одно или несколько изображений/серий;
- batch: directory или ZIP;
- отсутствие персональных данных в metadata не должно мешать inference.

## Outputs
- `anatomical_region`: `Поясничный отдел позвоночника` / `Проксимальный отдел бедра`;
- `quality_prob`: probability in [0,1];
- `quality_class`;
- `violation_type`;
- processing status/time;
- per-target probabilities;
- expert probabilities;
- `axis_angle_deg` для spine;
- `review_required`;
- optional saliency.

## Architecture
1. Projection gate.
2. Region-specific 3-member CNN ensemble.
3. Task-specific expert members for rare labels.
4. Soft-voting deep + expert.
5. Thresholded multi-label report.

## Expert members
- spine axis: geometric centerline angle around the vertical axis with 5° reference criterion;
- spine placement: coverage/FOV classifier;
- spine artifact: HOG + ExtraTrees;
- hip positioning/rotation: HOG + LogisticRegression, exported as the organizer-defined `Некорректная укладка` when triggered;
- hip ROI: border/coverage classifier.

## Data and validation
Supplied training archive: 100 studies, 499 DICOM. Labels are study-level. The main baseline split is study-level 80/20. Rare labels are strongly imbalanced.

Internal validation numbers are for engineering comparison only. They do not represent the organizer's hidden final score.

## Limitations
- tiny rare-class support, especially placement/ROI;
- no independently validated clinical performance;
- current projection gate is dataset-aware geometry logic;
- hip side assignment is not guaranteed;
- keypoint/segmentation metrics are not part of current model;
- DICOM SR / human-confirmed auto-correction are not implemented as mandatory production functions.

## Safety behavior
- failures return explicit `processing_status=Failure`;
- borderline or model-disagreement cases can be marked `review_required=true`;
- service is local and has no external inference dependency.

## Advanced ensemble update (2026-09-28)

The release adds a compact-feature LightGBM bagger behind the same public output contract. For each of the seven target heads, three study-level fold models plus one full-data model are stored locally (28 additional members total). Features include two grayscale downsampled views, compact HOG, LBP, intensity-grid statistics, Sobel statistics and thresholded shape summaries.

Engineering OOF evidence for the LightGBM bagger is documented in `reports/advanced_ensemble_lgbm_metrics.json`. These scores are not clinical validation. A combined evaluation using full-data members would be optimistic and is therefore not reported as an independent test result.

The external output format remains unchanged.
