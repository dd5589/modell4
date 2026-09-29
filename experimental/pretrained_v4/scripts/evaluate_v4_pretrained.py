#!/usr/bin/env python3
"""Evaluate trained v0.4 pretrained members on a fixed holdout.

This script deliberately does not change production blend weights. It produces
per-backbone and equal-weight ensemble probabilities for human/model selection.
"""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from dexa_ai.labels import build_manifest, SPINE_TARGETS, HIP_TARGETS
from dexa_ai.dicomio import read_dicom
from dexa_ai.preprocess import to_pil, letterbox
from dexa_ai.pretrained_models import build_pretrained, imagenet_normalize

REGION_TARGETS={'spine':SPINE_TARGETS,'hip':HIP_TARGETS}
IMAGE_SIZE=224

def study_split(rows, region, seed=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    reg=[r for r in rows if r.get('region')==region and 'error' not in r]
    studies=sorted({r['study'] for r in reg})
    y=[]; ss=[]
    for s in studies:
        v=next((r.get(f'{region}_quality') for r in reg if r['study']==s and r.get(f'{region}_quality') is not None),None)
        if v is not None: ss.append(s); y.append(int(v))
    ss=np.array(ss); y=np.array(y)
    tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(ss,y))
    return set(ss[tr]),set(ss[va])

def one_image(path):
    d=read_dicom(path); pil=letterbox(to_pil(d.array),IMAGE_SIZE); x=torch.from_numpy(np.asarray(pil,dtype=np.float32)/255.).unsqueeze(0)
    return d,x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--labels-xlsx',required=True); ap.add_argument('--weights',default='artifacts'); ap.add_argument('--region',choices=['spine','hip'],required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    rows=build_manifest(args.data_root,labels_xlsx=args.labels_xlsx); _,va=study_split(rows,args.region)
    targets=REGION_TARGETS[args.region]; models=[]
    for bk in ['resnet50','densenet121','efficientnet_b0','convnext_tiny']:
        wp=Path(args.weights)/'pretrained'/args.region/f'{bk}.pt'
        if not wp.exists(): continue
        ck=torch.load(wp,map_location='cpu',weights_only=False); m=build_pretrained(bk,len(targets),pretrained=False); m.load_state_dict(ck['state_dict']); m.eval(); models.append((bk,m))
    if not models: raise SystemExit('No trained v0.4 weights found')
    study_probs=defaultdict(lambda: {bk:[] for bk,_ in models}); gt={}
    for r in rows:
        if r.get('region')!=args.region or r['study'] not in va: continue
        d,x=one_image(r['path']); xt=imagenet_normalize(x)
        for bk,m in models:
            with torch.no_grad():
                p=torch.sigmoid(m(xt)).numpy()[0]
                pf=torch.sigmoid(m(torch.flip(xt,[3]))).numpy()[0]
                study_probs[r['study']][bk].append((p+pf)/2)
        gt[r['study']]={t:r.get(f'{args.region}_{t}') for t in targets}
    out={'region':args.region,'validation_studies':len(va),'backbones':[bk for bk,_ in models],'targets':{}}
    for ti,t in enumerate(targets):
        ids=[s for s in va if gt.get(s,{}).get(t) is not None and s in study_probs]
        if not ids: continue
        y=np.array([int(gt[s][t]) for s in ids])
        out['targets'][t]={}
        for bk,_ in models:
            p=np.array([max(np.asarray(study_probs[s][bk])[:,ti]) for s in ids])
            item={}
            try:item['roc_auc']=float(roc_auc_score(y,p))
            except:item['roc_auc']=None
            item['f1_at_0_5']=float(f1_score(y,(p>=0.5).astype(int),zero_division=0))
            try:item['pr_auc']=float(average_precision_score(y,p))
            except:item['pr_auc']=None
            out['targets'][t][bk]=item
        ens=np.mean([np.array([max(np.asarray(study_probs[s][bk])[:,ti]) for s in ids]) for bk,_ in models],axis=0)
        item={'f1_at_0_5':float(f1_score(y,(ens>=0.5).astype(int),zero_division=0))}
        try:item['roc_auc']=float(roc_auc_score(y,ens))
        except:item['roc_auc']=None
        try:item['pr_auc']=float(average_precision_score(y,ens))
        except:item['pr_auc']=None
        out['targets'][t]['equal_weight_ensemble']=item
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(args.out,'w',encoding='utf8'),ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
