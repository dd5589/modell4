# Release status

## Production submission package

Current production predictor: **advanced heterogeneous ensemble v3**.

### PASS in authoring/QA environment

- frozen artifact integrity
- model weight loading
- deterministic inference
- real DICOM parsing
- single-image API smoke test
- ZIP batch smoke test
- output contract validation
- organizer vocabulary validation
- hard-case generation
- Windows 10 local QA workflow

### Target-host gates

- Docker build on the target H200 VM
- CUDA/H200 runtime acceptance benchmark
- performance evidence on the target configuration
- organizer hidden closed-test evaluation

These gates cannot be truthfully marked as completed without access to the actual target H200/closed-test environment.

## Experimental next-generation model

`experimental/pretrained_v4/` contains the training pipeline for:

- ResNet-50
- DenseNet-121
- EfficientNet-B0
- ConvNeXt-Tiny

It is **not enabled in production inference** until real target-host training and honest study-level OOF/holdout comparison demonstrate benefit.
