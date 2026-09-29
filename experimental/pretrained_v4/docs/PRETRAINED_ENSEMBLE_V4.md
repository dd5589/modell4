# DXA AI v0.4 — Pretrained Heterogeneous Ensemble

## Цель

Добавить четыре независимых ImageNet-pretrained vision backbone к текущему v0.3 stack, не меняя внешний контракт API/CSV:

- ResNet-50
- DenseNet-121
- EfficientNet-B0
- ConvNeXt-Tiny

TorchVision предоставляет официальные ImageNet weights для этих архитектур; в обучающем скрипте используются именно `WeightsEnum.DEFAULT`/указанные ImageNet-1K варианты. Для DXA grayscale изображение повторяется на 3 канала и нормализуется по ImageNet mean/std. citeturn852259search6turn852259search7turn852259search8

## Почему это не просто «больше моделей»

v0.3 в основном использовал маленькие CNN + HOG/LightGBM/экспертные признаки. v0.4 добавляет более ёмкие feature extractors с разной архитектурой. Их роль — дать ансамблю независимые representations, а не увеличивать размер модели ради размера.

## Training recipe

1. Study-level train/validation split фиксируется одним seed.
2. Pretrained backbone сначала замораживается на 2 эпохи; обучается только multi-label head.
3. Затем размораживается поздний semantic stage backbone и используется малый LR.
4. Из-за дисбаланса используются per-target `pos_weight`.
5. Augmentations: horizontal flip, ±3° rotation, лёгкий intensity jitter; вертикальные flip запрещены.
6. Checkpoint выбирается по mean target-level F1 на validation.
7. Для production ensemble сохраняются frozen state dict + thresholds.
8. На target H200 запускать с CUDA AMP, pinned memory и 4 DataLoader workers.

## Important honesty rule

Если веса ImageNet невозможно скачать, `--no-pretrained` допустим **только для smoke-test**. Such a run must not be called a pretrained result and must not be included in final metrics.

## Command on H200

```bash
cd /app-or-project-root
bash scripts/prepare_h200_pretrained.sh \
  /data/train/Исследования \
  /data/train/разметка.xlsx \
  artifacts \
  12
```

The script downloads public pretrained weights through TorchVision if they are not already cached, then trains both regions.

## Expected artifacts

```text
artifacts/pretrained/spine/resnet50.pt
artifacts/pretrained/spine/densenet121.pt
artifacts/pretrained/spine/efficientnet_b0.pt
artifacts/pretrained/spine/convnext_tiny.pt
artifacts/pretrained/hip/resnet50.pt
artifacts/pretrained/hip/densenet121.pt
artifacts/pretrained/hip/efficientnet_b0.pt
artifacts/pretrained/hip/convnext_tiny.pt
```

## Acceptance criteria before enabling in production

A pretrained member enters the production blend only if:

- it completes all real-DICOM checks;
- no training/validation study leakage exists;
- it has finite F1/ROC-AUC where the target has both classes;
- it does not regress the current v0.3 visible-test output contract;
- its addition is evaluated on the same fixed study-level holdout as v0.3.

The external CSV stays unchanged:

```text
path_to_study
study_uid
image_uid
anatomical_region
quality_class
violation_type
processing_status
time_of_processing
quality_prob
```
