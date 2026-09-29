#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, time, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from dexa_ai.labels import build_manifest, SPINE_TARGETS, HIP_TARGETS
from dexa_ai.dicomio import read_dicom
from dexa_ai.preprocess import to_pil, letterbox
from dexa_ai.pretrained_models import (
    available_backbones, build_pretrained, freeze_backbone,
    imagenet_normalize, unfreeze_last_stage,
)

REGION_TARGETS = {'spine': SPINE_TARGETS, 'hip': HIP_TARGETS}
SEED = 20260928
IMAGE_SIZE = 224
BATCH_CUDA = 16
BATCH_CPU = 4


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_col(region: str, target: str) -> str:
    return f'{region}_{target}'


def study_split(rows, region: str, seed: int = SEED):
    targets = REGION_TARGETS[region]
    reg = [r for r in rows if r.get('region') == region and 'error' not in r]
    studies = sorted({r['study'] for r in reg if any(r.get(target_col(region,t)) is not None for t in targets)})
    yq = []
    usable = []
    for s in studies:
        vals = [r.get(target_col(region,'quality')) for r in reg if r['study']==s and r.get(target_col(region,'quality')) is not None]
        if vals:
            usable.append(s); yq.append(int(vals[0]))
    usable = np.array(usable); yq=np.array(yq)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(usable, yq))
    return set(usable[tr_idx]), set(usable[va_idx])


class DicomDataset(Dataset):
    def __init__(self, rows, region: str, train: bool, cache_path: str | None = None):
        self.rows = [r for r in rows if r.get('region')==region and 'error' not in r]
        self.region = region; self.train=train
        self.targets=REGION_TARGETS[region]
        self.cache_path = Path(cache_path) if cache_path else None
        self.images=[]
        self.labels=[]
        if self.cache_path and self.cache_path.exists():
            obj=torch.load(self.cache_path, map_location='cpu')
            self.images=obj['images']; self.labels=obj['labels']; self.studies=obj['studies']; self.paths=obj['paths']
        else:
            self.studies=[]; self.paths=[]
            for r in self.rows:
                arr = read_dicom(r['path']).array
                pil = letterbox(to_pil(arr), IMAGE_SIZE)
                x = torch.from_numpy(np.asarray(pil, dtype=np.float32)/255.0).unsqueeze(0)
                self.images.append(x)
                self.studies.append(r['study']); self.paths.append(r['path'])
                y=[]; m=[]
                for t in self.targets:
                    v=r.get(target_col(region,t)); y.append(0.0 if v is None else float(v)); m.append(0.0 if v is None else 1.0)
                self.labels.append(torch.tensor([y,m], dtype=torch.float32))
            self.labels=torch.stack(self.labels)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'images':self.images,'labels':self.labels,'studies':self.studies,'paths':self.paths}, self.cache_path)
    def subset(self, studies: set[str]) -> 'DatasetView':
        idx=[i for i,s in enumerate(self.studies) if s in studies]
        return DatasetView(self,idx,self.train)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self._item(idx)
    def _item(self, idx):
        x=self.images[idx].clone()
        if self.train:
            if torch.rand(()) < 0.5: x=torch.flip(x,[2])
            angle=float(torch.empty(1).uniform_(-3.0,3.0))
            try:
                import torchvision.transforms.functional as TF
                x=TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            except Exception:
                pass
            # mild intensity augmentation, preserving physical grayscale ordering
            if torch.rand(()) < 0.35:
                gain=float(torch.empty(1).uniform_(0.90,1.10)); bias=float(torch.empty(1).uniform_(-0.04,0.04))
                x=(x*gain+bias).clamp(0,1)
        x=imagenet_normalize(x.unsqueeze(0)).squeeze(0)
        y,m=self.labels[idx][0],self.labels[idx][1]
        return x,y,m,self.studies[idx],self.paths[idx]


class DatasetView(Dataset):
    def __init__(self, base: DicomDataset, indices: list[int], train: bool): self.base=base; self.indices=indices; self.train=train
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        idx=self.indices[i]; old=self.base.train; self.base.train=self.train
        out=self.base._item(idx); self.base.train=old; return out


def pos_weights(view, n_targets):
    pos=np.zeros(n_targets,float); total=np.zeros(n_targets,float)
    for i in view.indices:
        y,m=view.base.labels[i][0].numpy(), view.base.labels[i][1].numpy()
        pos += y*m; total += m
    neg=total-pos
    w=np.clip(neg/np.maximum(pos,1.0),1.0,15.0)
    return torch.tensor(w,dtype=torch.float32)


def study_probabilities(probs, study_ids):
    d=defaultdict(list)
    for p,s in zip(probs,study_ids): d[s].append(p)
    return {s: np.max(np.asarray(ps),axis=0) for s,ps in d.items()}


def choose_threshold(y,p):
    best=(0.5,-1.0)
    for t in np.linspace(0.05,0.95,91):
        f=f1_score(y,(p>=t).astype(int),zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best[0]


def evaluate(probs, study_ids, labels_by_study, targets):
    sp=study_probabilities(probs, study_ids); out={}; thresholds={}
    for ti,t in enumerate(targets):
        ids=[s for s in sp if labels_by_study.get(s,{}).get(t) is not None]
        y=np.array([labels_by_study[s][t] for s in ids],int); p=np.array([sp[s][ti] for s in ids],float)
        thr=choose_threshold(y,p) if len(np.unique(y))>1 else 0.5
        pred=(p>=thr).astype(int)
        item={'n':len(y),'positives':int(y.sum()),'threshold':thr,'f1':float(f1_score(y,pred,zero_division=0))}
        try:item['roc_auc']=float(roc_auc_score(y,p))
        except Exception:item['roc_auc']=float('nan')
        try:item['pr_auc']=float(average_precision_score(y,p))
        except Exception:item['pr_auc']=float('nan')
        thresholds[t]=thr; out[t]=item
    valid=[v['f1'] for v in out.values() if math.isfinite(v['f1'])]
    return {'targets':out,'macro_f1':float(np.mean(valid)) if valid else 0.0}, thresholds


def train_one(model, train_view, val_view, device, epochs, freeze_epochs, lr_head, lr_backbone, seed, out_path, targets):
    seed_all(seed); model.to(device)
    freeze_backbone(model)
    posw=pos_weights(train_view,len(targets)).to(device)
    loss_fn=nn.BCEWithLogitsLoss(reduction='none',pos_weight=posw)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr_head,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    loader=DataLoader(train_view,batch_size=BATCH_CUDA if device.type=='cuda' else BATCH_CPU,shuffle=True,num_workers=4 if device.type=='cuda' else 0,pin_memory=device.type=='cuda')
    best=-1; best_thr=None; best_state=None; patience=0
    for epoch in range(1,epochs+1):
        if epoch==freeze_epochs+1:
            unfreeze_last_stage(model)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr_backbone,weight_decay=1e-4)
        model.train(); losses=[]
        for x,y,m,_,_ in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); m=m.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                z=model(x); loss=(loss_fn(z,y)*m).sum()/m.sum().clamp_min(1.0)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        val_probs=[]; val_studies=[]
        model.eval()
        with torch.no_grad():
            vloader=DataLoader(val_view,batch_size=BATCH_CUDA if device.type=='cuda' else BATCH_CPU,shuffle=False,num_workers=4 if device.type=='cuda' else 0,pin_memory=device.type=='cuda')
            for x,_,_,s,_ in vloader:
                with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                    p=torch.sigmoid(model(x.to(device,non_blocking=True))).float().cpu().numpy()
                val_probs.append(p); val_studies.extend(list(s))
        val_probs=np.concatenate(val_probs,axis=0)
        ev,thr=evaluate(val_probs,val_studies,{s:{t:next(v for r in val_view.base.rows if r['study']==s for v in [r.get(target_col(val_view.base.region,t))]) for t in targets} for s in set(val_studies)},targets)
        score=ev['macro_f1']
        print(f'epoch={epoch} loss={np.mean(losses):.4f} val_macro_f1={score:.4f}',flush=True)
        if score>best:
            best=score; best_thr=thr; patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
            if patience>=3 and epoch>=freeze_epochs+2: break
    model.load_state_dict(best_state)
    torch.save({'state_dict':model.state_dict(),'backbone':model.name,'targets':targets,'image_size':IMAGE_SIZE,'best_macro_f1':best,'thresholds':best_thr},out_path)
    return best,best_thr


def collect_labels(rows,region):
    targets=REGION_TARGETS[region]; d={}
    for r in rows:
        if r.get('region')!=region: continue
        if r['study'] not in d: d[r['study']]={}
        for t in targets:
            v=r.get(target_col(region,t));
            if v is not None: d[r['study']][t]=int(v)
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--labels-xlsx',required=True)
    ap.add_argument('--out-dir',default='artifacts')
    ap.add_argument('--region',choices=['spine','hip','both'],default='both')
    ap.add_argument('--backbones',default='all',help='comma-separated or all')
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--freeze-epochs',type=int,default=2)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--no-pretrained',action='store_true')
    ap.add_argument('--max-studies',type=int,default=0,help='smoke-test limiter')
    args=ap.parse_args(); seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision('high')
    rows=build_manifest(args.data_root,labels_xlsx=args.labels_xlsx)
    regions=['spine','hip'] if args.region=='both' else [args.region]
    bks=available_backbones() if args.backbones=='all' else [x.strip() for x in args.backbones.split(',')]
    for b in bks:
        if b not in available_backbones(): raise ValueError(f'Unknown backbone {b}')
    summary={'version':'0.4-pretrained-ensemble','device':str(device),'pretrained':not args.no_pretrained,'backbones':bks,'regions':regions,'runs':[]}
    for region in regions:
        train_studies,val_studies=study_split(rows,region,args.seed)
        if args.max_studies:
            train_studies=set(sorted(train_studies)[:args.max_studies]); val_studies=set(sorted(val_studies)[:max(2,args.max_studies//4 or 2)])
        cache=Path(args.out_dir)/'pretrained_cache'/f'{region}_224.pt'
        ds=DicomDataset(rows,region,train=True,cache_path=str(cache))
        tv=ds.subset(train_studies); vv=ds.subset(val_studies)
        targets=REGION_TARGETS[region]
        region_dir=Path(args.out_dir)/'pretrained'/region; region_dir.mkdir(parents=True,exist_ok=True)
        print('region',region,'train_studies',len(train_studies),'val_studies',len(val_studies),'images',len(ds),'device',device,flush=True)
        for bk in bks:
            t0=time.time(); out=region_dir/f'{bk}.pt'
            model=build_pretrained(bk,len(targets),pretrained=not args.no_pretrained)
            best,thr=train_one(model,tv,vv,device,args.epochs,args.freeze_epochs,5e-4,3e-5,args.seed+hash(bk)%1000,str(out),targets)
            meta={'region':region,'backbone':bk,'out_path':str(out),'best_macro_f1':best,'thresholds':thr,'train_studies':len(train_studies),'val_studies':len(val_studies),'pretrained':not args.no_pretrained,'elapsed_sec':time.time()-t0}
            json.dump(meta,open(region_dir/f'{bk}.json','w'),indent=2,ensure_ascii=False); summary['runs'].append(meta); del model
        json.dump({'version':'0.4-pretrained-ensemble','region':region,'backbones':bks,'targets':targets,'input_size':IMAGE_SIZE,'imagenet_normalization':True,'notes':'Weights downloaded by torchvision on target host when --no-pretrained is absent.'},open(region_dir.parent/f'{region}_ensemble.json','w'),indent=2,ensure_ascii=False)
    out=Path(args.out_dir)/'pretrained_training_summary.json'; json.dump(summary,open(out,'w'),indent=2,ensure_ascii=False); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
