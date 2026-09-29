from __future__ import annotations
import os, tempfile, zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from .predictor import EnsemblePredictor
from .batch import process

DEFAULT_WEIGHTS=str(Path(__file__).resolve().parents[2]/'artifacts')
WEIGHTS=os.environ.get('DEXA_WEIGHTS',DEFAULT_WEIGHTS)
app=FastAPI(title='DXA Quality AI',version='0.1.0')
predictor=EnsemblePredictor(WEIGHTS) if Path(WEIGHTS).exists() else None

@app.get('/health')
def health():
    if predictor is None:
        return {'status':'degraded','model_loaded':False}
    return {
        'status':'ok',
        'model_loaded':True,
        'device':str(predictor.device),
        'deep_members':len(predictor.model_names),
        'expert_members':sorted(predictor.expert_models),
        'lgbm_members': {r: {t: len(v) for t, v in predictor.lgbm_models.get(r, {}).items()} for r in predictor.targets},
    }

@app.post('/predict')
async def predict(file:UploadFile=File(...), explain: bool=Query(False)):
    global predictor
    if predictor is None: raise HTTPException(503,'Model weights are not loaded')
    suffix=Path(file.filename or 'study.dcm').suffix or '.dcm'
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
        f.write(await file.read()); p=f.name
    try: return predictor.predict_one(p,explain=explain)
    finally: os.unlink(p)

@app.post('/predict-zip')
async def predict_zip(file:UploadFile=File(...), format: str=Query('json', pattern='^(json|csv)$')):
    with tempfile.TemporaryDirectory() as td:
        zp=Path(td)/'input.zip'; zp.write_bytes(await file.read())
        try:
            rows=process(str(zp),WEIGHTS)
        except zipfile.BadZipFile:
            raise HTTPException(400,'Uploaded file is not a valid ZIP archive')
        except Exception as e:
            raise HTTPException(500,f'Batch processing failed: {type(e).__name__}: {e}')
        if format=='json': return {'results':rows}
        out=Path(td)/'results.csv'
        with open(out,'w',encoding='utf8',newline='') as f:
            import csv
            fields=['path_to_study','study_uid','image_uid','anatomical_region','quality_class','violation_type','processing_status','time_of_processing','quality_prob']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        return Response(content=out.read_bytes(), media_type='text/csv', headers={'Content-Disposition':'attachment; filename=results.csv'})

@app.get('/',response_class=HTMLResponse)
def index():
    p=Path(__file__).parents[2]/'static'/'index.html'
    return p.read_text(encoding='utf8')
