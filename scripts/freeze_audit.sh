#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

printf '[1/5] compileall\n'
python -m compileall -q src tests

printf '[2/5] deep weights load + finite forward\n'
python - <<'PY'
from pathlib import Path
import torch
from dexa_ai.models import build_model
root=Path('artifacts/weights')
for region, targets in [('spine',['quality','placement','axis','artifact']),('hip',['quality','rotation','roi'])]:
    for name in ['fastcnn','dilatedcnn','depthwisecnn']:
        ck=torch.load(root/region/f'{name}.pt',map_location='cpu')
        m=build_model(name,len(targets)); m.load_state_dict(ck['state_dict'],strict=True); m.eval()
        y=m(torch.zeros(1,1,160,160))
        assert y.shape==(1,len(targets)) and torch.isfinite(y).all()
print('PASS')
PY

printf '[3/5] expert artifact load\n'
python - <<'PY'
from pathlib import Path
import pickle,json
root=Path('artifacts/experts')
for name in ['spine_placement','spine_artifact','hip_rotation','hip_roi']:
    with open(root/f'{name}.pkl','rb') as f: pickle.load(f)
with open(root/'expert_config.json',encoding='utf-8') as f: json.load(f)
print('PASS')
PY

printf '[4/5] deterministic forward\n'
python - <<'PY'
import numpy as np, torch
from dexa_ai.models import build_model
from dexa_ai.preprocess import tensor_from_array
arr=np.zeros((512,512),np.float32); arr[120:390,240:270]=1
x=tensor_from_array(arr).unsqueeze(0)
m=build_model('fastcnn',4).eval()
with torch.no_grad(): a=m(x); b=m(x)
assert torch.equal(a,b)
print('PASS')
PY

printf '[5/5] transient-clean hash regeneration\n'
find . -type f ! -path './SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt >/tmp/dexa_hashcheck.txt
printf 'PASS (%s files)\n' "$(wc -l < SHA256SUMS.txt)"
