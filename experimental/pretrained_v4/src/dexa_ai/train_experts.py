from __future__ import annotations
import argparse, json, time
from .experts import fit_experts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--labels-xlsx')
    ap.add_argument('--out-dir', default='artifacts/experts')
    args = ap.parse_args()
    t=time.perf_counter()
    cfg=fit_experts(args.data_root,args.labels_xlsx,args.out_dir)
    print(json.dumps(cfg,ensure_ascii=False,indent=2))
    print('elapsed_sec',round(time.perf_counter()-t,2))

if __name__ == '__main__': main()
