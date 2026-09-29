#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile, tempfile
from pathlib import Path
from dexa_ai.labels import build_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training-zip',required=True); ap.add_argument('--out-dir',default='data/prepared'); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    persistent = out / 'training_root'
    if persistent.exists():
        import shutil; shutil.rmtree(persistent)
    persistent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(a.training_zip) as z: z.extractall(td)
        import shutil
        # Copy the outer archive contents to a persistent staging root so the
        # manifest never contains dead TemporaryDirectory paths.
        for src in Path(td).iterdir():
            dst = persistent / src.name
            if src.is_dir(): shutil.copytree(src, dst)
            else: shutil.copy2(src, dst)
        import re
        def decode_component(name):
            return re.sub(r'#U([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1),16)), name)

        # The participant upload may wrap NД_для_обучения.zip and Для теста.zip
        # inside an outer Датасет.zip. Recursively unpack a candidate training zip
        # when the DICOM tree is not present at the first level.
        roots=list(persistent.rglob('Исследования'))
        if not roots:
            decoded=[]
            for d in persistent.rglob('*'):
                if d.is_dir() and decode_component(d.name) == 'Исследования':
                    decoded.append(d)
            roots=decoded
        if not roots:
            nested=[]
            for zpath in persistent.rglob('*.zip'):
                name=zpath.name.lower()
                if 'обуч' in name or 'train' in name or 'нд_' in name:
                    nested.append(zpath)
            if not nested:
                nested=list(persistent.rglob('*.zip'))
            if nested:
                nested_td=persistent/'_nested_training'
                nested_td.mkdir(exist_ok=True)
                with zipfile.ZipFile(nested[0]) as z: z.extractall(nested_td)
                roots=list(nested_td.rglob('Исследования'))
                if not roots:
                    roots=[d for d in nested_td.rglob('*') if d.is_dir() and decode_component(d.name) == 'Исследования']
        if not roots: raise SystemExit('Исследования folder not found inside archive (including nested ZIPs)')
        root=roots[0].parent
        xls=next(iter(root.glob('*.xlsx')),None)
        if xls is None:
            xls=next(iter(root.rglob('*.xlsx')),None)
        if xls is None: raise SystemExit('Annotation XLSX not found')
        build_manifest(str(root),str(xls),str(out/'manifest.csv'))
    print(out/'manifest.csv')

if __name__=='__main__': main()
