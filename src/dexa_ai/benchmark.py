from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from .dicomio import read_dicom
from .predictor import EnsemblePredictor

def sync(device):
    if getattr(device,'type','')=='cuda':
        import torch
        torch.cuda.synchronize(device)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--weights',default='artifacts')
    ap.add_argument('--device',default=None)
    ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--n',type=int,default=100)
    ap.add_argument('--out-json',default='benchmark.json')
    a=ap.parse_args()
    files=sorted(Path(a.input).rglob('*.dcm'))
    if not files: raise SystemExit('No DICOM files found')
    p=EnsemblePredictor(a.weights,device=a.device)
    warm=files[:min(a.warmup,len(files))]
    for f in warm: p.predict_one(str(f))
    times=[]
    n=min(a.n,len(files))
    for f in files[:n]:
        sync(p.device); t=time.perf_counter(); p.predict_one(str(f)); sync(p.device); times.append(time.perf_counter()-t)
    arr=np.asarray(times)*1000
    mean_ms=float(arr.mean())
    report={
        'device':str(p.device),
        'n':int(n),
        'p50_ms':float(np.percentile(arr,50)),
        'p95_ms':float(np.percentile(arr,95)),
        'mean_ms':mean_ms,
        'max_ms':float(arr.max()),
        'throughput_images_per_sec':float(1000/mean_ms),
        'estimated_3_image_study_mean_ms':float(3*mean_ms),
        'estimated_3_image_study_p95_ms':float(3*np.percentile(arr,95)),
    }
    json.dump(report,open(a.out_json,'w',encoding='utf8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
