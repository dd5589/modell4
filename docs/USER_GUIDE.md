# User guide

## Web interface

Start:

```bash
PYTHONPATH=src uvicorn dexa_ai.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`.

Upload one DICOM or a ZIP archive with DICOM files. The service returns a structured result with the organizer-defined anatomical region, binary quality class, optional violation types, processing status/time and `quality_prob`.

## Organizer output vocabulary

`anatomical_region`:
- `Поясничный отдел позвоночника`
- `Проксимальный отдел бедра`

Spine `violation_type` values:
- `Некорректная укладка`
- `Не выравнена ось позвоночника`
- `Присутствуют посторонние предметы`

Hip `violation_type` values:
- `Некорректная укладка`
- `Некорректная область интереса`

Multiple violations are separated by `; `; empty means no violation. Hip rotation is used internally as part of placement/positioning quality and is not emitted as a separate violation value.

`quality_prob` is always in `[0,1]`.

## Scale-dependent rules

The organizer confirmed a single scanner with pixel size 0.60 mm along X and 1.05 mm along Y. Therefore centimetre-based rules are evaluated from this fixed acquisition scale rather than requiring PixelSpacing DICOM tags.

## CLI batch

```bash
PYTHONPATH=src python -m dexa_ai.batch \
  --input /data/test.zip \
  --weights artifacts \
  --out-csv results.csv \
  --workers 1
```

## Interpreting rare-label outputs

`expert_probabilities` and `axis_angle_deg` are supporting signals. `review_required=true` indicates a near-threshold or deep-vs-expert disagreement case and should be treated as a human-in-the-loop review trigger, not an automatic clinical decision.
