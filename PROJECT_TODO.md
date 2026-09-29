# Project status / final checklist

## Done
- [x] DICOM reader
- [x] spine + proximal femur region detection
- [x] binary quality output
- [x] multi-label violation output
- [x] 3-member deep ensemble
- [x] spine HOG member
- [x] rare-label experts for placement/artifact/rotation/ROI
- [x] deterministic spine axis expert tied to 5° criterion
- [x] study-level validation artifacts
- [x] FastAPI `/health`, `/predict`, `/predict-zip`
- [x] batch CSV output
- [x] saliency explanation path
- [x] hard-case demo assets
- [x] README/model card/deployment/training/user docs
- [x] Dockerfile + compose
- [x] CPU benchmark
- [x] target-host H200 runbook

## Pending on target infrastructure
- [ ] `docker build` on target Linux host
- [ ] `docker run --gpus all` health check on target H200 host
- [ ] H200 latency / throughput measurement
- [ ] hidden closed-test batch run (not accessible to participants) and final submission CSV

## Optional / partial
- [ ] DICOM SR
- [ ] true segmentation/keypoint masks
- [ ] automatic correction with explicit specialist confirmation

## Do not do in the final 72h
- [ ] do not replace the architecture with a large transformer
- [ ] do not use image-level random split
- [ ] do not call debug data an independent test benchmark
- [ ] do not claim clinical validation from this small internal set

## Organizer-contract QA
- [x] exact anatomical_region vocabulary
- [x] exact violation_type vocabulary
- [x] hip rotation treated as internal placement subcriterion, not exported violation
- [x] quality_prob exported in [0,1]
- [x] fixed scanner scale documented: X=0.60 mm, Y=1.05 mm
