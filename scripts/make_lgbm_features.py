import sys, numpy as np, cv2
from pathlib import Path
from skimage import feature, transform, filters, morphology
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from dexa_ai.labels import build_manifest
from dexa_ai.dicomio import read_dicom
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--labels-xlsx',default=None); ap.add_argument('--out-features',required=True); ap.add_argument('--out-paths',required=True); args=ap.parse_args()
ROOT=args.data_root; XLSX=args.labels_xlsx
rows=build_manifest(ROOT,labels_xlsx=XLSX)
def feat(a):
 a=np.asarray(a,np.float32); lo,hi=np.percentile(a,[1,99]); hi=max(hi,lo+1e-3); x=np.clip((a-lo)/(hi-lo),0,1); x=cv2.resize(x,(160,192),interpolation=cv2.INTER_AREA)
 p=[cv2.resize(x,(32,32),interpolation=cv2.INTER_AREA).ravel(), cv2.resize(x,(16,16),interpolation=cv2.INTER_AREA).ravel()]
 p.append(feature.hog((x*255).astype(np.uint8),orientations=6,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys'))
 lbp=feature.local_binary_pattern((x*255).astype(np.uint8),8,2,method='uniform'); p.append(np.histogram(lbp,bins=np.arange(0,11),range=(0,10),density=True)[0])
 stats=[]
 for gy,gx in [(4,4),(8,8)]:
  for iy in range(gy):
   for ix in range(gx):
    q=x[iy*192//gy:(iy+1)*192//gy,ix*160//gx:(ix+1)*160//gx]
    stats += [q.mean(),q.std(),np.percentile(q,90),(q>.8).mean()]
 p.append(np.asarray(stats,np.float32))
 sob=filters.sobel(x); p.append(np.asarray([sob.mean(),sob.std(),np.percentile(sob,90),np.mean(np.abs(x-np.fliplr(x)))],np.float32))
 for th in (.6,.8,.9):
  m=x>th
  if m.any():
   yy,xx=np.nonzero(m); p.append(np.asarray([m.mean(),xx.mean()/160,yy.mean()/192,(xx.max()-xx.min()+1)/160,(yy.max()-yy.min()+1)/192],np.float32))
  else:p.append(np.zeros(5,np.float32))
 return np.concatenate([np.asarray(z,dtype=np.float32).ravel() for z in p])
F=np.stack([feat(read_dicom(r['path']).array) for r in rows]); print(F.shape); np.save(args.out_features,F)
np.save(args.out_paths,np.array([r['path'] for r in rows]))
