from __future__ import annotations
import os,json,argparse,time,random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from .labels import build_manifest,SPINE_TARGETS,HIP_TARGETS
from .dicomio import read_dicom
from .preprocess import tensor_from_array
from .hog_member import hog_features

def run(data_root,out_dir,region):
    rows=build_manifest(data_root,out_csv=os.path.join(out_dir,f'manifest_{region}_hog.csv')); allr=[r for r in rows if r['region']==region]
    labeled={s for s in set(r['study'] for r in allr) if any(r['study']==s and r[f'{region}_quality'] is not None for r in allr)}
    reg=[r for r in allr if r['study'] in labeled]; targets=SPINE_TARGETS if region=='spine' else HIP_TARGETS
    tensors={r['path']:tensor_from_array(read_dicom(r['path']).array,False) for r in reg}; X=hog_features([tensors[r['path']] for r in reg])
    studies=sorted(labeled); q=np.array([int(next(r[f'{region}_quality'] for r in reg if r['study']==s and r[f'{region}_quality'] is not None)) for s in studies])
    tr_i,va_i=next(StratifiedShuffleSplit(1,test_size=.2,random_state=42).split(studies,q)); tr_st=set(np.array(studies)[tr_i]); va_st=set(np.array(studies)[va_i]); tr=[r for r in reg if r['study'] in tr_st]; va=[r for r in reg if r['study'] in va_st]
    idx_tr=[reg.index(r) for r in tr]; idx_va=[reg.index(r) for r in va]
    metrics={}; th={}
    for ti,t in enumerate(targets):
        c=f'{region}_{t}'; ids=[i for i in idx_tr if reg[i][c] is not None]; y=np.array([int(reg[i][c]) for i in ids])
        clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=.1).fit(X[ids],y); pv=clf.predict_proba(X[idx_va])[:,1]
        by=defaultdict(list)
        for r,p in zip(va,pv): by[r['study']].append(float(p))
        ss=[s for s in studies if s in by and any(r['study']==s and r[c] is not None for r in reg)]; yy=np.array([int(next(r[c] for r in reg if r['study']==s and r[c] is not None)) for s in ss]); pp=np.array([max(by[s]) for s in ss])
        best=.5
        if len(np.unique(yy))>1: best=float(max((f1_score(yy,pp>=t0,zero_division=0),t0) for t0 in np.linspace(.1,.9,17))[1])
        th[t]=best; metrics[t]={'f1':float(f1_score(yy,pp>=best,zero_division=0)),'roc_auc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))>1 else None,'pr_auc':float(average_precision_score(yy,pp)) if len(np.unique(yy))>1 else None}
    model_dir=os.path.join(out_dir,'weights','hog',region); os.makedirs(model_dir,exist_ok=True)
    for t in targets:
        c=f'{region}_{t}'; ids=[i for i,r in enumerate(reg) if r[c] is not None]; y=np.array([int(reg[i][c]) for i in ids]); clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=.1).fit(X[ids],y)
        import pickle
        with open(os.path.join(model_dir,f'{t}.pkl'),'wb') as f: pickle.dump(clf,f,pickle.HIGHEST_PROTOCOL)
    result={'region':region,'metrics':metrics,'thresholds':th,'n_studies':len(studies),'n_images':len(reg)}; json.dump(result,open(os.path.join(out_dir,f'metrics_{region}_hog.json'),'w'),ensure_ascii=False,indent=2); print(json.dumps(result,ensure_ascii=False,indent=2)); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--region',choices=['spine','hip'],required=True); run(**vars(ap.parse_args()))
if __name__=='__main__': main()
