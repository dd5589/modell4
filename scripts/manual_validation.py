from __future__ import annotations
import argparse, base64, csv, html, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from dexa_ai.labels import build_manifest
from dexa_ai.predictor import EnsemblePredictor
from dexa_ai.dicomio import read_dicom

REGION_NAME = {
    'spine': 'Поясничный отдел позвоночника',
    'hip': 'Проксимальный отдел бедра',
}
SPINE_VIOLATIONS = {
    'placement': 'Некорректная укладка',
    'axis': 'Не выравнена ось позвоночника',
    'artifact': 'Присутствуют посторонние предметы',
}
HIP_VIOLATIONS = {
    'rotation': 'Некорректная укладка',
    'roi': 'Некорректная область интереса',
}


def as_int(v):
    if v is None or v == '':
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def study_gt(rows, region):
    # Labels are study-level and repeated over images in labels.py.
    r = rows[0]
    if region == 'spine':
        q = as_int(r.get('spine_quality'))
        vals = {k: as_int(r.get(f'spine_{k}')) for k in ('placement','axis','artifact')}
    else:
        q = as_int(r.get('hip_quality'))
        vals = {k: as_int(r.get(f'hip_{k}')) for k in ('rotation','roi')}
    violations = [SPINE_VIOLATIONS[k] if region == 'spine' else HIP_VIOLATIONS[k] for k,v in vals.items() if v == 1]
    return q, vals, violations


def sample_studies(rows, region, n, seed):
    grouped = defaultdict(list)
    for r in rows:
        if r.get('region') == region:
            grouped[r['study']].append(r)
    studies = sorted(grouped)
    rng = random.Random(seed)
    # Positive-first for QA usefulness, then fill with normal studies.
    positives, negatives, unknown = [], [], []
    for s in studies:
        q, vals, _ = study_gt(grouped[s], region)
        if q == 1:
            positives.append(s)
        elif q == 0:
            negatives.append(s)
        else:
            unknown.append(s)
    rng.shuffle(positives); rng.shuffle(negatives); rng.shuffle(unknown)
    selected = (positives + negatives + unknown)[:n]
    return selected, grouped


def png_data_uri(arr: np.ndarray, max_side=900):
    x = np.asarray(arr, dtype=np.float32)
    x -= np.nanmin(x)
    mx = float(np.nanmax(x))
    if mx > 0:
        x /= mx
    img = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))
    scale = min(1.0, max_side / max(img.size))
    if scale < 1:
        img = img.resize((max(1, int(img.width*scale)), max(1, int(img.height*scale))), Image.Resampling.LANCZOS)
    from io import BytesIO
    b = BytesIO(); img.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode('ascii')


def run(args):
    rows = build_manifest(args.data_root, args.labels_xlsx)
    selected, grouped = sample_studies(rows, args.region, args.n, args.seed)
    pred = EnsemblePredictor(args.weights)
    out = []
    html_rows = []
    violation_allowed = set(SPINE_VIOLATIONS.values()) if args.region == 'spine' else set(HIP_VIOLATIONS.values())

    for study in selected:
        gt_q, gt_vals, gt_v = study_gt(grouped[study], args.region)
        for r in grouped[study]:
            try:
                d = read_dicom(r['path'])
                p = pred.predict_one(r['path'])
                pred_q = as_int(p.get('quality_class'))
                pred_v = [x.strip() for x in (p.get('violation_type') or '').split(';') if x.strip()]
                exact_v = sorted(pred_v) == sorted(gt_v)
                axis_rule_ok = ''
                if args.region == 'spine':
                    angle = p.get('axis_angle_deg')
                    if angle is not None:
                        axis_pred = 'Не выравнена ось позвоночника' in pred_v
                        axis_rule_ok = bool((float(angle) > 5.0) == axis_pred)
                success = (gt_q is None or pred_q == gt_q) and all(v in violation_allowed for v in pred_v)
                row = {
                    'study': study,
                    'path': r['path'],
                    'anatomical_region_gt': REGION_NAME[args.region],
                    'anatomical_region_pred': p.get('anatomical_region',''),
                    'quality_gt': gt_q,
                    'quality_pred': pred_q,
                    'quality_prob': p.get('quality_prob'),
                    'violation_gt': '; '.join(gt_v),
                    'violation_pred': '; '.join(pred_v),
                    'violations_exact_match': exact_v,
                    'quality_exact_match': (gt_q is None or pred_q == gt_q),
                    'contract_region_ok': p.get('anatomical_region') in REGION_NAME.values(),
                    'contract_violation_ok': all(v in violation_allowed for v in pred_v),
                    'axis_rule_ok': axis_rule_ok,
                    'axis_angle_deg': p.get('axis_angle_deg',''),
                    'review_required': p.get('review_required', False),
                    'processing_status': p.get('processing_status'),
                    'time_sec': p.get('time_of_processing'),
                }
                out.append(row)
                html_rows.append((row, png_data_uri(d.array)))
            except Exception as e:
                out.append({'study':study,'path':r['path'],'error':f'{type(e).__name__}: {e}'})

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv,'w',newline='',encoding='utf-8') as f:
        fields = sorted({k for r in out for k in r})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)

    head = f'''<!doctype html><html><head><meta charset="utf-8"><title>DXA Manual Gold Audit</title>
    <style>body{{font-family:Arial,sans-serif;margin:20px}} .case{{border:1px solid #bbb;margin:18px 0;padding:12px}} img{{max-width:520px;max-height:520px;border:1px solid #ddd}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:5px;vertical-align:top}} .bad{{background:#ffe6e6}} .ok{{background:#eaffea}} pre{{white-space:pre-wrap}}</style></head><body>
    <h1>DXA Manual Gold Audit — {html.escape(args.region)}</h1>
    <p>Selected studies: {len(selected)}; DICOM images checked: {len(out)}.</p>'''
    blocks=[]
    for row, uri in html_rows:
        bad = not all(bool(row.get(k, True)) for k in ('quality_exact_match','contract_region_ok','contract_violation_ok')) or row.get('axis_rule_ok') is False
        cls='bad' if bad else 'ok'
        cells=[]
        for k in ('study','path','quality_gt','quality_pred','quality_prob','violation_gt','violation_pred','violations_exact_match','quality_exact_match','axis_angle_deg','axis_rule_ok','review_required','processing_status','time_sec'):
            cells.append(f'<tr><th>{html.escape(k)}</th><td>{html.escape(str(row.get(k,'')))}</td></tr>')
        blocks.append(f'<div class="case {cls}"><h2>{html.escape(str(row["study"]))}</h2><img src="{uri}"><table>{"".join(cells)}</table></div>')
    Path(args.out_html).write_text(head + ''.join(blocks) + '</body></html>', encoding='utf-8')
    summary = {
        'selected_studies': len(selected),
        'images_checked': len(out),
        'quality_exact_matches': sum(bool(r.get('quality_exact_match')) for r in out),
        'violation_exact_matches': sum(bool(r.get('violations_exact_match')) for r in out),
        'contract_region_failures': sum(not bool(r.get('contract_region_ok', True)) for r in out),
        'contract_violation_failures': sum(not bool(r.get('contract_violation_ok', True)) for r in out),
        'axis_rule_failures': sum(r.get('axis_rule_ok') is False for r in out),
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'CSV:  {args.out_csv}')
    print(f'HTML: {args.out_html}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--labels-xlsx')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--region', choices=['spine','hip'], required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-html', required=True)
    ap.add_argument('--out-json', required=True)
    args=ap.parse_args(); run(args)

if __name__=='__main__': main()
