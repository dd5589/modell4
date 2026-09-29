from __future__ import annotations
import os,json,argparse,time,random,warnings
from collections import defaultdict
import numpy as np, torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from .dicomio import read_dicom
from .preprocess import tensor_from_array
from .models import build_model
from .labels import build_manifest,SPINE_TARGETS,HIP_TARGETS
from .metrics import binary_metrics

MODELS=['fastcnn','dilatedcnn','depthwisecnn']; SEEDS=[17,31,73]; BATCH=32

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def col(region,t): return f'{region}_{t}'
def pos_weights(records,targets,region):
    w=[]
    for t in targets:
        vals=[int(r[col(region,t)]) for r in records if r[col(region,t)] is not None]; pos=sum(vals); neg=len(vals)-pos
        w.append(max(1.0,min(10.0,neg/max(1,pos))))
    return torch.tensor(w,dtype=torch.float32)

def fit(model,records,tensors,targets,region,device,epochs,seed):
    seed_all(seed); model.to(device); pw=pos_weights(records,targets,region).to(device)
    loss_fn=nn.BCEWithLogitsLoss(reduction='none',pos_weight=pw); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for _ in range(epochs):
        model.train(); order=np.random.permutation(len(records))
        for j in range(0,len(order),BATCH):
            ids=order[j:j+BATCH]; x=torch.stack([tensors[records[i]['path']] for i in ids]).to(device)
            if len(ids):
                flip=torch.rand(x.size(0),device=device)<.25
                if flip.any(): x[flip]=torch.flip(x[flip],dims=[3])
            y=torch.tensor([[0 if records[i][col(region,t)] is None else int(records[i][col(region,t)]) for t in targets] for i in ids],dtype=torch.float32,device=device)
            m=torch.tensor([[0 if records[i][col(region,t)] is None else 1 for t in targets] for i in ids],dtype=torch.float32,device=device)
            opt.zero_grad(set_to_none=True); z=model(x); loss=(loss_fn(z,y)*m).sum()/m.sum().clamp_min(1); loss.backward(); opt.step()
    return model

def predict(model,records,tensors,device):
    model.eval(); out=[]
    with torch.no_grad():
        for j in range(0,len(records),64):
            x=torch.stack([tensors[records[i]['path']] for i in range(j,min(j+64,len(records)))]).to(device); out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out)

def run(data_root,out_dir,region,epochs=10):
    os.makedirs(out_dir,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': torch.set_num_threads(min(8,torch.get_num_threads()))
    rows=build_manifest(data_root,out_csv=os.path.join(out_dir,f'manifest_{region}.csv')); reg_all=[r for r in rows if r['region']==region]
    labeled_studies={s for s in {r['study'] for r in reg_all} if any(r['study']==s and r[col(region,'quality')] is not None for r in reg_all)}
    reg=[r for r in reg_all if r['study'] in labeled_studies]
    tensors={}
    for i,r in enumerate(reg):
        if r['path'] not in tensors: tensors[r['path']]=tensor_from_array(read_dicom(r['path']).array,False)
    studies=sorted(set(r['study'] for r in reg)); quality=[]
    for s in studies:
        vals=[r[col(region,'quality')] for r in reg if r['study']==s and r[col(region,'quality')] is not None]; quality.append(int(vals[0]))
    studies=np.array(studies); quality=np.array(quality)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42); tr_i,va_i=next(sss.split(studies,quality)); tr_st=set(studies[tr_i]); va_st=set(studies[va_i])
    tr=[r for r in reg if r['study'] in tr_st]; va=[r for r in reg if r['study'] in va_st]
    print(region,'studies',len(studies),'images',len(reg),'train',len(tr_st),'val',len(va_st),flush=True)
    oof={t:[] for t in SPINE_TARGETS if region=='spine'} if region=='spine' else {t:[] for t in HIP_TARGETS}
    study_probs={m:{t:{} for t in oof} for m in MODELS}
    for mi,mname in enumerate(MODELS):
        print('val model',mname,flush=True); model=fit(build_model(mname,len(oof)),tr,tensors,list(oof.keys()),region,device,max(5,epochs//2),SEEDS[mi]); p=predict(model,va,tensors,device)
        for ti,t in enumerate(oof):
            by=defaultdict(list)
            for r,prob in zip(va,p[:,ti]): by[r['study']].append(float(prob))
            for s,ps in by.items(): study_probs[mname][t][s]=max(ps)
        del model
    thresholds={}; metrics={}
    for t in oof:
        ids=[s for s in studies if any(r['study']==s and r[col(region,t)] is not None for r in reg)]
        y=np.array([int(next(r[col(region,t)] for r in reg if r['study']==s and r[col(region,t)] is not None)) for s in ids])
        p=np.array([np.mean([study_probs[m][t].get(s,.5) for m in MODELS]) for s in ids])
        best=.5
        if len(np.unique(y))>1:
            from sklearn.metrics import f1_score
            fs=[(f1_score(y,(p>=t).astype(int),zero_division=0),t) for t in np.linspace(.1,.9,17)]; best=float(max(fs)[1])
        thresholds[t]=best; metrics[t]=binary_metrics(y,p,best,bootstrap=300)
    final_dir=os.path.join(out_dir,'weights',region); os.makedirs(final_dir,exist_ok=True)
    for mi,mname in enumerate(MODELS):
        print('final model',mname,flush=True); model=fit(build_model(mname,len(oof)),reg,tensors,list(oof.keys()),region,device,epochs,SEEDS[mi]); torch.save({'model_name':mname,'state_dict':model.cpu().state_dict(),'targets':list(oof.keys()),'image_size':160},os.path.join(final_dir,mname+'.pt')); del model
    result={'region':region,'n_studies':len(studies),'n_images':len(reg),'validation_studies':len(va_st),'thresholds':thresholds,'metrics':metrics,'models':MODELS,'device':str(device),'epochs':epochs}
    json.dump(result,open(os.path.join(out_dir,f'metrics_{region}.json'),'w'),ensure_ascii=False,indent=2)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--region',choices=['spine','hip'],required=True); ap.add_argument('--epochs',type=int,default=10); a=ap.parse_args(); t=time.time();
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); r=run(a.data_root,a.out_dir,a.region,a.epochs)
    print(json.dumps(r,ensure_ascii=False,indent=2)); print('elapsed_sec',round(time.time()-t,2),flush=True)
if __name__=='__main__': main()
