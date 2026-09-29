from __future__ import annotations
import os,json,argparse,warnings,random
from collections import defaultdict
import numpy as np, torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from .labels import build_manifest,SPINE_TARGETS,HIP_TARGETS
from .dicomio import read_dicom
from .preprocess import tensor_from_array
from .models import build_model
from .train_region import fit,predict
from .hog_member import hog_features
from .metrics import binary_metrics
from skimage.feature import hog

def run(data_root,out_dir,region,epochs=5):
    rows=build_manifest(data_root); allr=[r for r in rows if r['region']==region]
    labeled={s for s in set(r['study'] for r in allr) if any(r['study']==s and r[f'{region}_quality'] is not None for r in allr)}
    reg=[r for r in allr if r['study'] in labeled]
    targets=SPINE_TARGETS if region=='spine' else HIP_TARGETS
    tensors={r['path']:tensor_from_array(read_dicom(r['path']).array,False) for r in reg}
    studies=sorted(labeled); q=np.array([int(next(r[f'{region}_quality'] for r in reg if r['study']==s and r[f'{region}_quality'] is not None)) for s in studies])
    tr_i,va_i=next(StratifiedShuffleSplit(1,test_size=.2,random_state=42).split(studies,q)); tr_st=set(np.array(studies)[tr_i]); va_st=set(np.array(studies)[va_i]); tr=[r for r in reg if r['study'] in tr_st]; va=[r for r in reg if r['study'] in va_st]
    deep=[]
    for mi,name in enumerate(['fastcnn','dilatedcnn','depthwisecnn']):
        model=fit(build_model(name,len(targets)),tr,tensors,targets,region,torch.device('cpu'),epochs,[17,31,73][mi]); deep.append(predict(model,va,tensors,torch.device('cpu'))); del model
    deep_p=np.mean(deep,axis=0)
    X=np.stack([hog((tensors[r['path']][0].numpy()*255).astype('uint8'),orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys') for r in reg])
    idx_tr=[reg.index(r) for r in tr]; idx_va=[reg.index(r) for r in va]
    hp=np.zeros((len(va),len(targets)))
    for ti,t in enumerate(targets):
        c=f'{region}_{t}'; ids=[i for i in idx_tr if reg[i][c] is not None]; y=np.array([int(reg[i][c]) for i in ids]); clf=__import__('sklearn').linear_model.LogisticRegression(max_iter=2000,class_weight='balanced',C=.1).fit(X[ids],y); hp[:,ti]=clf.predict_proba(X[idx_va])[:,1]
    dw,hw=({'spine':(.5,.5),'hip':(1.0,0.0)})[region]; p=dw*deep_p+hw*hp
    metrics={}; thresholds={}
    for ti,t in enumerate(targets):
        c=f'{region}_{t}'; by=defaultdict(list)
        for r,prob in zip(va,p[:,ti]): by[r['study']].append(float(prob))
        ss=[s for s in studies if s in by and any(r['study']==s and r[c] is not None for r in reg)]
        y=np.array([int(next(r[c] for r in reg if r['study']==s and r[c] is not None)) for s in ss]); pp=np.array([max(by[s]) for s in ss]);
        th=.5 if len(np.unique(y))<2 else max(((f1_score(y,pp>=t0,zero_division=0),t0) for t0 in np.linspace(.1,.9,17)), key=lambda z:z[0])[1]
        thresholds[t]=float(th); metrics[t]=binary_metrics(y,pp,thresholds[t],bootstrap=1000)
    def sanitize(obj):
        if isinstance(obj, dict): return {k:sanitize(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)): return [sanitize(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj): return None
        return obj
    result={'region':region,'n_studies':len(studies),'n_images':len(reg),'validation_studies':len(va_st),'deep_weight':dw,'hog_weight':hw,'thresholds':thresholds,'metrics':sanitize(metrics),'models':['fastcnn','dilatedcnn','depthwisecnn'],'split':'stratified 80/20 by study'}
    os.makedirs(out_dir,exist_ok=True); json.dump(result,open(os.path.join(out_dir,f'metrics_deployed_{region}.json'),'w'),ensure_ascii=False,indent=2); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--region',choices=['spine','hip'],required=True); ap.add_argument('--epochs',type=int,default=5); a=ap.parse_args();
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); run(a.data_root,a.out_dir,a.region,a.epochs)
