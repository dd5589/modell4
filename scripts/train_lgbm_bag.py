from __future__ import annotations
import os,sys,json,pickle,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from lightgbm import LGBMClassifier,early_stopping
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from dexa_ai.labels import build_manifest
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--labels-xlsx',default=None); ap.add_argument('--features',required=True); ap.add_argument('--out-dir',required=True); args=ap.parse_args()
ROOT=args.data_root; XLSX=args.labels_xlsx
rows=build_manifest(ROOT,labels_xlsx=XLSX); X=np.load(args.features,mmap_mode='r')
OUT=Path(args.out_dir); OUT.mkdir(parents=True,exist_ok=True)
for region,targets in [('spine',['quality','placement','axis','artifact']),('hip',['quality','rotation','roi'])]:
 rr=[r for r in rows if r['region']==region]
 for t in targets:
  c=f'{region}_{t}'; studies=sorted({r['study'] for r in rr if r[c] is not None}); ystudy=np.array([int(next(r[c] for r in rr if r['study']==s and r[c] is not None)) for s in studies]); pos=int(ystudy.sum()); neg=int((ystudy==0).sum())
  if pos<2 or neg<2: continue
  n_splits=min(3,pos,neg); skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=2026); oof={}; models=[]
  print(region,t,'studies',len(studies),'pos',pos,'folds',n_splits,flush=True)
  for fold,(tri,vii) in enumerate(skf.split(studies,ystudy)):
   trst=set(np.array(studies)[tri]); vast=set(np.array(studies)[vii]); tr=[r for r in rr if r['study'] in trst and r[c] is not None]; va=[r for r in rr if r['study'] in vast and r[c] is not None]
   ti=np.array([rows.index(r) for r in tr]); vi=np.array([rows.index(r) for r in va]); y=np.array([int(r[c]) for r in tr]);
   cnt=defaultdict(int)
   for r in tr: cnt[r['study']]+=1
   sw=np.array([1/cnt[r['study']] for r in tr],np.float32)
   m=LGBMClassifier(n_estimators=100,max_depth=3,num_leaves=7,learning_rate=.05,subsample=.95,colsample_bytree=.5,reg_lambda=8,reg_alpha=.2,min_child_samples=5,verbosity=-1,random_state=2026+fold,n_jobs=4,class_weight='balanced')
   with warnings.catch_warnings(): warnings.simplefilter('ignore'); m.fit(X[ti],y,sample_weight=sw)
   path=OUT/f'{region}_{t}_fold{fold}.txt'; m.booster_.save_model(str(path)); models.append(path.name)
   pv=m.predict_proba(X[vi])[:,1]; by=defaultdict(list)
   for r,p in zip(va,pv): by[r['study']].append(float(p))
   for s,ps in by.items(): oof[s]=max(ps)
  ss=sorted(oof); yy=np.array([int(next(r[c] for r in rr if r['study']==s and r[c] is not None)) for s in ss]); pp=np.array([oof[s] for s in ss]); best=max(((f1_score(yy,pp>=th,zero_division=0),th) for th in np.linspace(.03,.97,95)),key=lambda z:z[0])[1]
  fr=[r for r in rr if r[c] is not None]; fi=np.array([rows.index(r) for r in fr]); yf=np.array([int(r[c]) for r in fr]); cnt=defaultdict(int); [cnt.__setitem__(r['study'],cnt[r['study']]+1) for r in fr]; sw=np.array([1/cnt[r['study']] for r in fr],np.float32)
  full=LGBMClassifier(n_estimators=100,max_depth=3,num_leaves=7,learning_rate=.05,subsample=.95,colsample_bytree=.5,reg_lambda=8,reg_alpha=.2,min_child_samples=5,verbosity=-1,random_state=2026,n_jobs=4,class_weight='balanced')
  with warnings.catch_warnings(): warnings.simplefilter('ignore'); full.fit(X[fi],yf,sample_weight=sw)
  fullpath=OUT/f'{region}_{t}_full.txt'; full.booster_.save_model(str(fullpath))
  metrics={'region':region,'target':t,'n_studies':len(studies),'positive_studies':pos,'n_folds':n_splits,'oof_f1':float(f1_score(yy,pp>=best,zero_division=0)),'oof_auc':float(roc_auc_score(yy,pp)),'oof_prauc':float(average_precision_score(yy,pp)),'threshold':float(best),'fold_models':models,'full_model':fullpath.name}
  json.dump(metrics,open(OUT/f'{region}_{t}_metrics.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
  print(region,t,'OOF F1',round(metrics['oof_f1'],3),'AUC',round(metrics['oof_auc'],3),'PR',round(metrics['oof_prauc'],3),'thr',round(best,3),flush=True)
