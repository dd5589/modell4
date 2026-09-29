from __future__ import annotations
from dataclasses import dataclass
import glob, os, csv, json
from collections import defaultdict
from .dicomio import read_dicom

SPINE_TARGETS=['quality','placement','axis','artifact']
HIP_TARGETS=['quality','rotation','roi']
TARGET_TO_DESCRIPTION={
 'placement':'Некорректная укладка',
 'axis':'Не выравнена ось позвоночника',
 'artifact':'Присутствуют посторонние предметы',
 'rotation':'Некорректная укладка',
 'roi':'Некорректная область интереса',
}

def build_manifest(train_root: str, labels_xlsx: str | None = None, out_csv: str | None = None):
    train_root=os.path.abspath(train_root)
    research=os.path.join(train_root,'Исследования')
    if not os.path.isdir(research):
        # Some competition archives contain UTF-8 path components serialized as #Uhhhh.
        import re
        def decode_component(name):
            return re.sub(r'#U([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1),16)), name)
        for child in os.listdir(train_root):
            if decode_component(child) == 'Исследования' and os.path.isdir(os.path.join(train_root,child)):
                research=os.path.join(train_root,child)
                break
    if labels_xlsx is None:
        xls=glob.glob(os.path.join(train_root,'*.xlsx'))+glob.glob(os.path.join(train_root,'*.XLSX'))
        if not xls: raise FileNotFoundError('Excel annotation file not found')
        labels_xlsx=xls[0]
    # Minimal XLSX reader: the annotation workbook is only read as raw cell values.
    import zipfile, xml.etree.ElementTree as ET
    NS={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    with zipfile.ZipFile(labels_xlsx) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('a:si',NS): shared.append(''.join(t.text or '' for t in si.iter('{%s}t'%NS['a'])))
        wbroot=ET.fromstring(z.read('xl/workbook.xml'))
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        relmap={rel.attrib['Id']:rel.attrib['Target'] for rel in rels}
        first=wbroot.find('a:sheets/a:sheet',NS)
        target=relmap[first.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
        sheet_path='xl/'+target.lstrip('/') if not target.startswith('xl/') else target
        sh=ET.fromstring(z.read(sheet_path))
    def col_idx(ref):
        letters=''.join(ch for ch in ref if ch.isalpha()); n=0
        for ch in letters: n=n*26+ord(ch.upper())-64
        return n-1
    grid={}
    for c in sh.findall('.//a:sheetData/a:row/a:c',NS):
        idx=col_idx(c.attrib['r']); v=c.find('a:v',NS); txt=''
        if v is not None:
            txt=v.text or ''
            if c.attrib.get('t')=='s': txt=shared[int(txt)]
        elif c.attrib.get('t')=='inlineStr':
            txt=''.join(t.text or '' for t in c.iter('{%s}t'%NS['a']))
        grid[idx]=grid.get(idx,{})
    matrix=[]
    for row_el in sh.findall('.//a:sheetData/a:row',NS):
        cells={}
        for c in row_el.findall('a:c',NS):
            idx=col_idx(c.attrib['r']); v=c.find('a:v',NS); txt=''
            if v is not None:
                txt=v.text or ''
                if c.attrib.get('t')=='s': txt=shared[int(txt)]
            elif c.attrib.get('t')=='inlineStr': txt=''.join(t.text or '' for t in c.iter('{%s}t'%NS['a']))
            try: val=float(txt) if txt and c.attrib.get('t')!='s' and '.' in txt else int(txt) if txt and c.attrib.get('t')!='s' else txt
            except: val=txt
            cells[idx]=val
        matrix.append([cells.get(i) for i in range(19)])
    labels={}
    for row in matrix[2:]:
        study=row[1]
        if not study: continue
        hip_rot=[v for v in (row[5],row[7]) if v is not None]
        hip_roi=[v for v in (row[6],row[8]) if v is not None]
        hip_q=[v for v in (row[10],row[11]) if v is not None]
        labels[str(study)]={
            'spine_quality':row[9], 'spine_placement':row[2], 'spine_axis':row[3], 'spine_artifact':row[4],
            'hip_quality':max(hip_q) if hip_q else None, 'hip_rotation':max(hip_rot) if hip_rot else None,
            'hip_roi':max(hip_roi) if hip_roi else None,
            'right_rotation':row[5],'right_roi':row[6],'left_rotation':row[7],'left_roi':row[8],
            'right_quality':row[10],'left_quality':row[11],'expert_comment':row[12] or ''
        }
    rows=[]
    for study in labels:
        folder=os.path.join(research,study)
        for p in glob.glob(folder+'/**/*.dcm',recursive=True):
            try:
                d=read_dicom(p)
                region='spine' if d.cols>=295 else 'hip' if 220<=d.cols<=294 else 'unknown'
                r={'study':study,'path':p,'region':region,'rows':d.rows,'cols':d.cols,'instance':d.instance_number,'study_uid':d.study_uid,'image_uid':d.sop_uid}
                for k,v in labels[study].items(): r[k]=v
                rows.append(r)
            except Exception as e:
                rows.append({'study':study,'path':p,'region':'unknown','rows':'','cols':'','instance':'','study_uid':'','image_uid':'','error':str(e),**labels[study]})
    if out_csv:
        with open(out_csv,'w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows)))); w.writeheader(); w.writerows(rows)
    return rows
