# Architecture

## Production predictor

```text
DICOM / ZIP
    ↓
DICOM decoding + normalization
    ↓
Anatomical region detection
    ↓
Region-specific inference
    ├── 3 CNN members
    ├── HOG / classical feature members
    ├── LightGBM bagging members
    └── task-specific experts
         ├── axis geometry
         ├── placement
         ├── artifact
         ├── hip rotation
         └── hip ROI
    ↓
Target-specific probability blending
    ↓
quality_prob
    ↓
quality_class
    ↓
violation_type
```

## External contract

The predictor deliberately keeps the competition output vocabulary stable. Internal model changes do not require changes to the API/CSV contract.

## Rare-label strategy

Rare violations are not delegated to a single neural classifier. Classical features and deterministic geometry are included as independent signals and can trigger `review_required` for human-in-the-loop review.
