from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .labels import build_manifest
from .dicomio import read_dicom
from .predictor import EnsemblePredictor
from .experts import centerline_angle_deg, resized

CASES_TARGETS=[
    ('axis','spine','spine_axis'),
    ('artifact','spine','spine_artifact'),
    ('roi','hip','hip_roi'),
]


def norm(a):
    a=np.asarray(a,float); finite=np.isfinite(a)
    if not finite.any(): return np.zeros(a.shape,dtype=np.uint8)
    v=a[finite]; lo,hi=np.percentile(v,[1,99]); x=np.clip((a-lo)/(hi-lo+1e-8),0,1)
    return (x*255).astype(np.uint8)

def first_positive_region(rows, region, truth_key):
    for r in rows:
        if r.get('region') != region:
            continue
        if truth_key == 'hip_roi':
            vals=[r.get('right_roi'),r.get('left_roi')]; vals=[v for v in vals if v in (0,1)]
            ok=max(vals) if vals else None
        else:
            ok=r.get(truth_key)
        if ok == 1:
            return r
    raise RuntimeError(f'No positive {truth_key} sample found in region={region}')

def decode_data_uri(uri):
    if not uri: return None
    return Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))).convert('RGB')

def truth_value(row, truth_key):
    if truth_key=='hip_roi':
        vals=[row.get('right_roi'),row.get('left_roi')]; vals=[int(v) for v in vals if v in (0,1)]; return max(vals) if vals else None
    return row.get(truth_key)

def make_case(kind, row, truth_key, prediction, out):
    d=read_dicom(row['path']); im=Image.fromarray(norm(d.array)).convert('RGB')
    im.thumbnail((640,640)); card=Image.new('RGB',(760,700),'black'); card.paste(im,((760-im.width)//2,25))
    draw=ImageDraw.Draw(card); font=ImageFont.load_default()
    image_x=(760-im.width)//2; image_y=25
    if kind=='axis':
        ang=centerline_angle_deg(d.array)
        # Draw an indicative centerline estimate over the image.
        w,h=im.size; cx=image_x+w//2; length=int(h*.35); rad=np.deg2rad(ang); dx=int(np.sin(rad)*length/2)
        y0=image_y+int(h*.32); y1=image_y+int(h*.68)
        draw.line((cx-dx,y0,cx+dx,y1),fill='white',width=4)
    elif kind=='roi':
        w,h=im.size; bx=int(w*.08); by=int(h*.08); draw.rectangle((image_x+bx,image_y+by,image_x+w-bx,image_y+h-by),outline='white',width=4)
    heat=decode_data_uri(prediction.get('saliency_base64'))
    if heat:
        heat.thumbnail((240,240)); card.paste(heat,(510,390))
    text=[f'case={kind}',f'study={row["study"]}',f'ground_truth={truth_key}:{truth_value(row,truth_key)}',f'quality_pred={prediction.get("quality_class")}',f'violations={prediction.get("violation_type","")}',f'probs={prediction.get("probabilities",{})}',f'review_required={prediction.get("review_required")}']
    if 'axis_angle_deg' in prediction: text.append(f'axis_angle_deg={prediction["axis_angle_deg"]:.2f}')
    y=560
    for line in text:
        draw.text((15,y),line,fill='white',font=font); y+=18
    card.save(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--weights',default='artifacts'); ap.add_argument('--out-dir',default='reports/hard_cases'); a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=build_manifest(a.data_root)
    p=EnsemblePredictor(a.weights,device='cpu')
    manifest=[]
    for kind,region,key in CASES_TARGETS:
        row=first_positive_region(rows,region,key)
        study=row['study']
        pred=p.predict_one(row['path'],explain=True)
        fn=out/f'case_{kind}.png'; make_case(kind,row,key,pred,fn)
        manifest.append({'kind':kind,'study':study,'region':region,'image_uid':pred.get('image_uid'),'ground_truth':truth_value(row,key),'prediction':pred,'file':str(fn)})
    json.dump(manifest,open(out/'hard_cases.json','w',encoding='utf8'),ensure_ascii=False,indent=2)
    print(json.dumps([{k:v for k,v in m.items() if k!='prediction'} for m in manifest],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
