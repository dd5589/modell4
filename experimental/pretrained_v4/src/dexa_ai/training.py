"""CLI wrapper for the reproducible two-region training pipeline."""
from __future__ import annotations
import argparse, subprocess, sys

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out-dir',default='artifacts'); ap.add_argument('--epochs',type=int,default=10)
    a=ap.parse_args()
    for region in ('spine','hip'):
        cmd=[sys.executable,'-m','dexa_ai.train_region','--data-root',a.data_root,'--out-dir',a.out_dir,'--region',region,'--epochs',str(a.epochs)]
        subprocess.run(cmd,check=True)
    for region in ('spine','hip'):
        cmd=[sys.executable,'-m','dexa_ai.train_hog','--data-root',a.data_root,'--out-dir',a.out_dir,'--region',region]
        subprocess.run(cmd,check=True)
if __name__=='__main__': main()
