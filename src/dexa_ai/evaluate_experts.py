from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .labels import build_manifest
from .dicomio import read_dicom
from .experts import normalize_array, placement_features, roi_features, vision_features, axis_probability
from .experts import _study_label, _make_model, _fit_weighted

TARGETS = ['spine_placement','spine_axis','spine_artifact','hip_rotation','hip_roi']


def eval_target(rows, cache, target, test_size=0.25, seed=42):
    target_rows=[r for r in rows if _study_label(r,target) is not None]
    by_study=defaultdict(list)
    labels={}
    for r in target_rows:
        by_study[r['study']].append(r)
        labels[r['study']]=int(_study_label(r,target))
    studies=np.asarray(sorted(labels))
    y=np.asarray([labels[s] for s in studies],dtype=int)
    if len(np.unique(y))<2 or min(np.bincount(y))<2:
        return {'status':'insufficient_class_support','n_studies':int(len(studies)),'positive_studies':int(y.sum())}
    sss=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=seed)
    tr_i, va_i=next(sss.split(studies,y))
    tr_st=set(studies[tr_i]); va_st=set(studies[va_i])
    tr=[r for r in target_rows if r['study'] in tr_st]
    va=[r for r in target_rows if r['study'] in va_st]
    ytr=np.asarray([int(_study_label(r,target)) for r in tr],dtype=int)
    feature_fn = placement_features if target=='spine_placement' else roi_features if target=='hip_roi' else vision_features
    if target=='spine_axis':
        pimg=np.asarray([axis_probability(cache[r['path']]) for r in va],dtype=float)
    else:
        Xtr=np.stack([feature_fn(cache[r['path']]) for r in tr])
        Xva=np.stack([feature_fn(cache[r['path']]) for r in va])
        model=_make_model(target); _fit_weighted(model,Xtr,ytr,[r['study'] for r in tr]); pimg=model.predict_proba(Xva)[:,1]
    by=defaultdict(list); yby=defaultdict(list)
    for r,p in zip(va,pimg): by[r['study']].append(float(p)); yby[r['study']].append(int(_study_label(r,target)))
    sy=np.asarray([max(yby[s]) for s in by],dtype=int)
    sp=np.asarray([max(by[s]) for s in by],dtype=float)
    pred=(sp>=0.5).astype(int)
    return {
        'status':'ok',
        'n_train_studies':int(len(tr_st)),
        'n_holdout_studies':int(len(sy)),
        'positive_train_studies':int(sum(labels[s] for s in tr_st)),
        'positive_holdout_studies':int(sy.sum()),
        'roc_auc':float(roc_auc_score(sy,sp)),
        'pr_auc':float(average_precision_score(sy,sp)),
        'f1@0.5':float(f1_score(sy,pred,zero_division=0)),
        'threshold':0.5,
        'note':'Target-specific stratified study-level split; use only as engineering evidence because rare positive support is small.'
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--labels-xlsx')
    ap.add_argument('--out',default='reports/expert_validation_stratified.json')
    args=ap.parse_args()
    rows=build_manifest(args.data_root,args.labels_xlsx)
    cache={}
    for r in rows:
        if r['path'] not in cache:
            cache[r['path']]=normalize_array(read_dicom(r['path']).array)
    result={t:eval_target([r for r in rows if r['region']==('spine' if t.startswith('spine_') else 'hip')],cache,t) for t in TARGETS}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    json.dump(result,open(args.out,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
