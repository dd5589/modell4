import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dexa_ai.dicomio import read_dicom


def test_read_training_dicom():
    """Smoke-test a DICOM only when the caller provides a real file.

    This keeps the test suite portable inside and outside the competition VM.
    """
    path = os.environ.get('DEXA_TEST_DICOM')
    if not path:
        pytest.skip('Set DEXA_TEST_DICOM to a DICOM file for data-backed smoke test')
    p = Path(path)
    assert p.exists(), p
    d = read_dicom(p)
    assert d.array.ndim == 2 and d.rows > 0 and d.cols > 0
    assert d.sop_uid
