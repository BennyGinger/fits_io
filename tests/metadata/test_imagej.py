import json

from fits_io.metadata.context import MetadataBuildContext
from fits_io.metadata.imagej import build_tiff_metadata
from fits_io.metadata.models import TiffMetadata
from fits_io.metadata.provenance import FITS_TAG


def _make_ctx() -> MetadataBuildContext:
    return MetadataBuildContext(n_channels=2, labels=['GFP', 'mCherry'], axes='TZCYX', base_payload={'existing': 'data'}, step_name='test_step', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))


def test_build_tiff_metadata_basic_fields():
    ctx = _make_ctx()
    payload = {'status': 'active', 'user_name': 'test_user', 'z_projection_method': None, 'test_step': {'dist': 'test-dist', 'version': 'x', 'timestamp': 'y'}}
    out = build_tiff_metadata(ctx, payload)
    assert isinstance(out, TiffMetadata)
    assert out.imagej_meta['axes'] == 'TZCYX'
    assert out.imagej_meta['finterval'] == 11.0
    assert out.imagej_meta['Labels'] == ['GFP', 'mCherry']
    assert out.imagej_meta['unit'] == 'micron'
    assert out.resolution == (2.0, 4.0)


def test_build_tiff_metadata_encodes_payload_in_extratags():
    ctx = _make_ctx()
    payload = {'status': 'skip', 'user_name': 'test_user', 'z_projection_method': 'mean', 'test_step': {'param': 'value'}}
    out = build_tiff_metadata(ctx, payload)
    assert out.extratags is not None
    tag, dtype, count, raw, writeonce = out.extratags[0]
    assert tag == FITS_TAG
    assert dtype == 'B'
    assert count == len(raw)
    assert writeonce is True
    decoded = json.loads(raw.decode('utf-8'))
    assert decoded['status'] == 'skip'
    assert decoded['test_step']['param'] == 'value'