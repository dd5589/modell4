# Target H200 + Closed Test Runbook

## A. H200 host acceptance

Prerequisites on each target VM: Linux x86_64, NVIDIA driver with `nvidia-smi`, Docker Engine, and NVIDIA Container Toolkit. The application itself runs entirely locally.

1. Copy `dexa_ai_release_final.zip` to the VM and extract it.
2. Verify GPU:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

3. Verify Docker GPU access:

```bash
docker run --rm --gpus all pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

4. Run one-command acceptance:

```bash
cd dexa_ai_final
bash scripts/h200_acceptance.sh /ABS/PATH/TO/Для\ теста.zip /ABS/PATH/TO/benchmark 200
```

5. The gate must produce `TARGET H200 ACCEPTANCE PASS`. Inspect:
`reports/h200_acceptance/health.json`, `h200_benchmark.json`, `batch_results.csv`, `acceptance_summary.json`.

6. Record the benchmark numbers in the final pitch and final QA workbook. Do not use CPU numbers as H200 numbers.

### If Docker is missing (Ubuntu)

Install Docker Engine from Docker's official repository, then install/configure NVIDIA Container Toolkit. NVIDIA's official sequence is `nvidia-ctk runtime configure --runtime=docker` followed by `systemctl restart docker`.

## B. Closed test

The hidden closed test is organizer-only; it is not in the participant package. Do not attempt to obtain or reconstruct it.

If the organizer/CI provides the closed dataset path to the container, run the exact same inference path used for visible testing:

```bash
cd dexa_ai_final
PYTHONPATH=src python -m dexa_ai.batch \
  --input /ABS/PATH/TO/CLOSED_TEST.zip \
  --weights artifacts \
  --out-csv closed_test_results.csv \
  --workers 1
```

Then validate:

```bash
python - <<'PY'
import csv
from pathlib import Path
p=Path('closed_test_results.csv')
rows=list(csv.DictReader(p.open(encoding='utf-8')))
required=['path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing','quality_prob']
assert rows and list(rows[0].keys())==required
regions={'Поясничный отдел позвоночника','Проксимальный отдел бедра'}
spine={'Некорректная укладка','Не выравнена ось позвоночника','Присутствуют посторонние предметы'}
hip={'Некорректная укладка','Некорректная область интереса'}
for r in rows:
    assert r['processing_status'] in {'Success','Failure'}
    assert r['anatomical_region'] in regions or r['processing_status']=='Failure'
    if r['quality_prob']:
        q=float(r['quality_prob']); assert 0<=q<=1
    if r['violation_type']:
        allowed=spine if r['anatomical_region'].startswith('Поясничный') else hip
        assert all(x in allowed for x in r['violation_type'].split('; '))
print('contract PASS; rows=',len(rows))
PY
```

If the organizer provides ground-truth labels after evaluation, compute sensitivity, specificity, balanced accuracy, F1, ROC-AUC/PR-AUC, macro-F1, and 95% confidence intervals. The participant does not claim a hidden-test score before the organizer publishes/calculates it.
