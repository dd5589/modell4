from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from .labels import build_manifest
from .dicomio import read_dicom
from .experts import normalize_array, placement_features, roi_features, vision_features, axis_probability, _study_label, _make_model, _fit_weighted

TARGETS=['spine_placement','spine_axis','spine_artifact','hip_rotation','hip_roi']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--labels-xlsx'); ap.add_argument('--out',default='artifacts/experts/expert_oof.json'); a=ap.parse_args()
    rows=build_manifest(a.data_root,a.labels_xlsx)
    cache={r['path']:normalize_array(read_dicom(r['path']).array) for r in rows}
    out={}
    for target in TARGETS:
        region='spine' if target.startswith('spine_') else 'hip'
        rr=[r for r in rows if r['region']==region and _study_label(r,target) is not None]
        by=defaultdict(list)
        for r in rr: by[r['study']].append(r)
        studies=np.array(sorted(by.keys()))
        y=np.array([int(_study_label(by[s][0],target)) for s in studies])
        # Keep folds feasible for rare labels; 3 folds is the minimum for ROI here.
        n_splits=min(5,int(y.sum()),int((1-y).sum()))
        skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
        p=np.full(len(studies),np.nan,float)
        for tr_idx,va_idx in skf.split(studies,y):
            tr_st=set(studies[tr_idx]); va_st=set(studies[va_idx])
            if target=='spine_axis':
                for j,s in zip(va_idx,studies[va_idx]): p[j]=max(axis_probability(cache[r['path']]) for r in by[s])
            else:
                fn=placement_features if target=='spine_placement' else roi_features if target=='hip_roi' else vision_features
                tr=[r for r in rr if r['study'] in tr_st]; va=[r for r in rr if r['study'] in va_st]
                Xtr=np.stack([fn(cache[r['path']]) for r in tr]); ytr=np.array([int(_study_label(r,target)) for r in tr])
                Xva=np.stack([fn(cache[r['path']]) for r in va])
                model=_make_model(target); _fit_weighted(model,Xtr,ytr,[r['study'] for r in tr]); pv=model.predict_proba(Xva)[:,1]
                byp=defaultdict(list)
                for r,pp in zip(va,pv): byp[r['study']].append(float(pp))
                for j,s in zip(va_idx,studies[va_idx]): p[j]=max(byp[s])
        candidates=[]
        for t in np.linspace(.05,.95,19): candidates.append((float(f1_score(y,(p>=t).astype(int),zero_division=0)),float(t)))
        best=max(candidates)
        out[target]={'n_studies':int(len(y)),'positive_studies':int(y.sum()),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'oof_best_f1':best[0],'oof_threshold':best[1],'positive_prob_median':float(np.median(p[y==1])),'negative_prob_median':float(np.median(p[y==0]))}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w',encoding='utf8'),ensure_ascii=False,indent=2); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
