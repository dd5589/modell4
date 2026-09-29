from __future__ import annotations
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog

def hog_features(tensor_batch):
    # tensor_batch: [N,1,160,160], values 0..1
    out=[]
    for x in tensor_batch:
        a=(x[0].numpy()*255).clip(0,255).astype('uint8')
        out.append(hog(a,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys'))
    return np.asarray(out,dtype=np.float32)

def train_hog_models(records,tensors,targets,region,out_dir,study_split=None):
    import os
    os.makedirs(out_dir,exist_ok=True)
    X=hog_features([tensors[r['path']] for r in records])
    models={}
    for t in targets:
        col=f'{region}_{t}'; ids=[i for i,r in enumerate(records) if r[col] is not None]
        y=np.array([int(records[i][col]) for i in ids]); clf=LogisticRegression(max_iter=2000,class_weight='balanced',C=.1).fit(X[ids],y); models[t]=clf
    for t,m in models.items():
        with open(os.path.join(out_dir,f'{t}.pkl'),'wb') as f: pickle.dump(m,f,protocol=pickle.HIGHEST_PROTOCOL)
    return models

def predict_hog(models,tensors,records):
    X=hog_features([tensors[r['path']] for r in records]); import numpy as np
    return np.column_stack([models[t].predict_proba(X)[:,1] for t in models])
