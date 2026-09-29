from __future__ import annotations
import argparse, csv, os, zipfile, tempfile, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .predictor import EnsemblePredictor

def collect_dicom(root):
    return sorted(Path(root).rglob('*.dcm'))

def process(input_path,weights,out_csv=None,max_workers=1):
    predictor=EnsemblePredictor(weights)
    tmp=None
    inp=Path(input_path)
    if inp.suffix.lower()=='.zip':
        tmp=tempfile.TemporaryDirectory();
        with zipfile.ZipFile(inp) as z: z.extractall(tmp.name)
        files=collect_dicom(tmp.name)
    else: files=collect_dicom(inp)
    rows=[]
    def one(p):
        try: return predictor.predict_one(str(p))
        except Exception as e: return {'path_to_study':str(p),'processing_status':'Failure','time_of_processing':0.0,'error_message':f'{type(e).__name__}: {e}'}
    if max_workers>1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex: rows=list(ex.map(one,files))
    else: rows=[one(p) for p in files]
    fields=['path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing','quality_prob']
    if out_csv:
        with open(out_csv,'w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    if tmp: tmp.cleanup()
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--weights',required=True); ap.add_argument('--out-csv'); ap.add_argument('--workers',type=int,default=1); a=ap.parse_args(); t=time.perf_counter(); rows=process(a.input,a.weights,out_csv=a.out_csv,max_workers=a.workers); print(f'processed={len(rows)} elapsed_sec={time.perf_counter()-t:.3f}')
if __name__=='__main__': main()
