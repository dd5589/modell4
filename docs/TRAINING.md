# Training / retraining

## 1. Dataset layout

Expected input:

```text
/data/train/
  Исследования/<study_uid>/**/*.dcm
  разметка.xlsx
```

The training code builds a study-level manifest. A split must always be performed by study, never by individual DICOM image, to avoid leakage between related images.

## 2. Deep ensemble

Three compact CNN members are trained per anatomical region:

- `fastcnn`
- `dilatedcnn`
- `depthwisecnn`

Each member predicts the region-specific targets. Spine also keeps a classical HOG/LogisticRegression member.

## 3. Rare-label experts

Because the rare labels have very few positive studies, additional interpretable experts are trained:

- `spine_placement`: field-of-view / coverage features;
- `spine_axis`: deterministic centerline angle with the explicit 5° criterion;
- `spine_artifact`: HOG + morphology + tree ensemble;
- `hip_rotation`: HOG + morphology + logistic head;
- `hip_roi`: border/coverage features + logistic head.

Train them with:

```bash
PYTHONPATH=src python -m dexa_ai.train_experts \
  --data-root /data/train \
  --labels-xlsx /data/train/разметка.xlsx \
  --out-dir artifacts/experts
```

## 4. Reproducibility

Pin Python package versions in `requirements.txt` / `requirements-cpu.txt`. Keep random seeds fixed in model training and expert heads. Store the exact manifest and configuration used for a submission.

## 5. Evaluation

Use study-level holdout/CV. Report F1, sensitivity, specificity, balanced accuracy, ROC-AUC and PR-AUC. For tiny rare-label classes, explicitly report positive support and confidence intervals and do not interpret one split as clinical validation.
