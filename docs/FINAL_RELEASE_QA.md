# Final release QA

## PASS before target-host acceptance
- Frozen 3-CNN regional ensemble and task-specific experts.
- Exact organizer output vocabulary.
- `quality_prob` added to CSV in `[0,1]`.
- Real DICOM single-image inference passes.
- Real DICOM ZIP batch passes: 3/3 Success on participant-visible debug set.
- `prepare_dataset.py` handles the `#Uhhhh` path serialization used by the supplied training archive.
- Study-level split and deterministic forward are frozen.
- Hard cases regenerated from current training archive: axis, artifact, ROI.
- Documentation, model card, source code, Dockerfile and one-command H200 acceptance script included.

## PENDING target host
- Docker build on Linux target machine.
- CUDA service health on H200.
- H200 p50/p95/throughput benchmark.
- Full closed-test batch on organizer-only hidden data.

## Optional / partial
- DICOM SR is not implemented.
- True keypoint/segmentation output is not implemented.
- Automatic correction with specialist confirmation is not implemented.

These optional functions are separated from the mandatory submission core.
