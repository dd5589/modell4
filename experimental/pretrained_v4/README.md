# Experimental pretrained ensemble v4

This is a training-only extension of the production repository.

It adds four pretrained vision backbones:

- ResNet-50
- DenseNet-121
- EfficientNet-B0
- ConvNeXt-Tiny

They are **not enabled by default in the production predictor**. They become production candidates only after training on the competition data and a leakage-safe study-level OOF/holdout comparison against the current production ensemble.

Start on the H200 target with:

```bash
bash scripts/h200_train_v4.sh \
  /data/train \
  /data/train/разметка.xlsx \
  /data/artifacts_v4 \
  12
```

The training pipeline downloads public pretrained weights through TorchVision when network access is available, or can use a pre-populated Torch cache.
