"""DICOM I/O with pydicom when available and a small native parser fallback.
The fallback supports the uncompressed little-endian CR DXA files used in the
provided dataset (MONOCHROME2, 8/16-bit PixelData).
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
from pathlib import Path
import numpy as np

LONG_VR = {'OB','OD','OF','OL','OV','OW','SQ','UC','UR','UT','UN'}
VALID_VR = set('AE AS AT CS DA DS DT FD FL IS LO LT OB OD OF OL OV PN SH SL SQ SS ST SV TM UC UI UL UN UR US UT UV'.split())

@dataclass
class DicomImage:
    array: np.ndarray
    study_uid: str
    series_uid: str
    sop_uid: str
    instance_number: str
    rows: int
    cols: int
    photometric: str = 'MONOCHROME2'
    metadata: dict | None = None

# ----------------------------- fallback parser -----------------------------
def _meta_end(b: bytes):
    o = 132
    ts = None
    while o + 8 <= len(b):
        g, e = struct.unpack_from('<HH', b, o)
        if g != 0x0002:
            break
        vr = b[o+4:o+6].decode('ascii', 'ignore')
        if vr in LONG_VR:
            l = struct.unpack_from('<I', b, o+8)[0]; vs = o + 12
        else:
            l = struct.unpack_from('<H', b, o+6)[0]; vs = o + 8
        if (g, e) == (0x0002, 0x0010):
            ts = b[vs:vs+l].rstrip(b'\0').decode('ascii', 'ignore')
        o = vs + l
    return o, ts

def _scan(b: bytes, start: int, explicit: bool):
    o, out = start, {}
    while o + 8 <= len(b):
        g, e = struct.unpack_from('<HH', b, o)
        if g == 0xfffe:
            break
        if explicit:
            vr = b[o+4:o+6].decode('ascii', 'ignore')
            if vr not in VALID_VR:
                break
            if vr in LONG_VR:
                if o + 12 > len(b): break
                l = struct.unpack_from('<I', b, o+8)[0]; vs = o + 12
            else:
                l = struct.unpack_from('<H', b, o+6)[0]; vs = o + 8
        else:
            vr = None; l = struct.unpack_from('<I', b, o+4)[0]; vs = o + 8
        if l == 0xffffffff:
            # Sequences are not required for image pixels. Skip to their delimiter.
            end = b.find(b'\xfe\xff\xdd\xe0', vs)
            if end < 0: break
            o = end + 8; continue
        if vs + l > len(b):
            break
        out[(g, e)] = (vr, b[vs:vs+l])
        o = vs + l
        if (g, e) == (0x7fe0, 0x0010):
            break
    return out

def _text(d, tag):
    raw = d.get(tag, (None, b''))[1]
    return raw.rstrip(b'\0 ').decode('utf-8', 'ignore')

def _u16(d, tag, default=None):
    raw = d.get(tag, (None, b''))[1]
    return struct.unpack_from('<H', raw[:2].ljust(2, b'\0'))[0] if len(raw) >= 2 else default

def _native_read(path: str | Path) -> DicomImage:
    b = Path(path).read_bytes()
    if len(b) < 132 or b[128:132] != b'DICM':
        raise ValueError('Unsupported DICOM file: missing DICM preamble')
    start, ts = _meta_end(b)
    if ts not in ('1.2.840.10008.1.2', '1.2.840.10008.1.2.1', None):
        raise ValueError(f'Unsupported Transfer Syntax for fallback parser: {ts}')
    d = _scan(b, start, explicit=(ts != '1.2.840.10008.1.2'))
    rows = _u16(d, (0x0028,0x0010)); cols = _u16(d, (0x0028,0x0011))
    bits_alloc = _u16(d, (0x0028,0x0100), 8); pixel_repr = _u16(d, (0x0028,0x0103), 0)
    if not rows or not cols:
        raise ValueError('DICOM Rows/Columns missing')
    raw = d.get((0x7fe0,0x0010), (None,b''))[1]
    if not raw:
        raise ValueError('DICOM PixelData missing')
    if bits_alloc == 8:
        arr = np.frombuffer(raw, dtype=np.uint8)
    elif bits_alloc == 16:
        arr = np.frombuffer(raw, dtype=np.int16 if pixel_repr else np.uint16)
    else:
        raise ValueError(f'Unsupported BitsAllocated={bits_alloc}')
    arr = arr[:rows*cols].reshape(rows, cols)
    return DicomImage(arr.astype(np.float32), _text(d,(0x0020,0x000d)), _text(d,(0x0020,0x000e)),
                      _text(d,(0x0008,0x0018)), _text(d,(0x0020,0x0013)), rows, cols,
                      _text(d,(0x0028,0x0004)) or 'MONOCHROME2',
                      {'TransferSyntaxUID':ts, 'SeriesDescription':_text(d,(0x0008,0x103e)),
                       'PatientOrientation':_text(d,(0x0020,0x0020))})

def read_dicom(path: str | Path) -> DicomImage:
    path = str(path)
    try:
        import pydicom  # type: ignore
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
            arr = arr.max() - arr
        slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
        arr = arr * slope + intercept
        return DicomImage(
            array=arr, study_uid=str(getattr(ds,'StudyInstanceUID','')),
            series_uid=str(getattr(ds,'SeriesInstanceUID','')),
            sop_uid=str(getattr(ds,'SOPInstanceUID','')), instance_number=str(getattr(ds,'InstanceNumber','')),
            rows=int(arr.shape[-2]), cols=int(arr.shape[-1]),
            photometric=str(getattr(ds,'PhotometricInterpretation','MONOCHROME2')),
            metadata={k: str(getattr(ds,k,'')) for k in ['SeriesDescription','PatientOrientation','BodyPartExamined','Laterality','Modality']}
        )
    except Exception:
        return _native_read(path)
