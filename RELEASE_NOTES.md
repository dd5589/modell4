# DXA Quality AI — Release Notes

## Release
`dexa-ai-freeze-2026-09-26-scale-v3`

## Closed before target host
- Frozen 3-CNN regional ensemble plus task-specific experts.
- Exact organizer output vocabulary enforced.
- `quality_prob` added to CSV output in [0,1].
- Hip rotation is an internal positioning criterion and is exported as `Некорректная укладка` when triggered.
- Organizer-confirmed scanner scale recorded: X=0.60 mm, Y=1.05 mm.
- Real DICOM `/predict` and `/predict-zip` smoke tests pass.
- Participant-visible `Для теста.zip`: 3/3 files processed successfully.
- Hard-case demos regenerated against the current training archive.
- README, Model Card, training/deployment/user docs, QA workbook and release pitch included.
- Source archive path encoding (`#Uhhhh`) is handled by `prepare_dataset.py` / manifest builder.

## Target-host gates still required
- Docker build on Linux target host.
- GPU-enabled `/health` on H200.
- H200 latency/throughput benchmark.
- Batch execution on the hidden organizer closed-test data.

The hidden closed test is not provided to participants, so no participant-side result is claimed for it.

## Advanced stack v0.3
- Added 28 LightGBM bagging members (3-fold + full model across 7 target heads).
- Added compact multi-scale grayscale/HOG/LBP/shape feature extractor.
- External CSV/API vocabulary and presentation design unchanged.
- Visible 3-file test remained 3/3 Success with the same predicted class/taxonomy outputs as the previous release.
- Engineering OOF metrics are documented separately; no independent clinical-validation claim is made.
