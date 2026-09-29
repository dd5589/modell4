from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

def binary_metrics(y_true,y_prob,thr=.5,bootstrap=1000,seed=42):
    y=np.asarray(y_true,dtype=int); p=np.asarray(y_prob,float); pred=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else float('nan'); spec=tn/(tn+fp) if tn+fp else float('nan')
    out={'f1':float(f1_score(y,pred,zero_division=0)), 'sensitivity':float(sens), 'specificity':float(spec)}
    try: out['roc_auc']=float(roc_auc_score(y,p))
    except: out['roc_auc']=float('nan')
    try: out['pr_auc']=float(average_precision_score(y,p))
    except: out['pr_auc']=float('nan')
    rng=np.random.default_rng(seed); n=len(y); vals={k:[] for k in ['f1','sensitivity','specificity','roc_auc','pr_auc']}
    for _ in range(bootstrap):
        idx=rng.integers(0,n,n)
        yy=y[idx]; pp=p[idx]; pr=(pp>=thr).astype(int)
        tn,fp,fn,tp=confusion_matrix(yy,pr,labels=[0,1]).ravel()
        s=tp/(tp+fn) if tp+fn else np.nan; sp=tn/(tn+fp) if tn+fp else np.nan
        try: a=roc_auc_score(yy,pp)
        except: a=np.nan
        try: pa=average_precision_score(yy,pp)
        except: pa=np.nan
        vals['f1'].append(f1_score(yy,pr,zero_division=0)); vals['sensitivity'].append(s); vals['specificity'].append(sp); vals['roc_auc'].append(a); vals['pr_auc'].append(pa)
    for k,arr in vals.items():
        arr=np.asarray(arr,float); arr=arr[np.isfinite(arr)]
        if len(arr): out[k+'_ci95']=[float(np.percentile(arr,2.5)),float(np.percentile(arr,97.5))]
    return out
