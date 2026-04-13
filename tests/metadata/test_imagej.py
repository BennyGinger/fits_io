import json

from fits_io.metadata.context import MetadataBuildContext
from fits_io.metadata.imagej import build_tiff_metadata
from fits_io.metadata.models import TiffMetadata
from fits_io.metadata.codec import FITS_TAG


def _make_ctx() -> MetadataBuildContext:
    return MetadataBuildContext(n_channels=2, labels=['GFP', 'mCherry'], axes='TZCYX', base_payload={'existing': 'data'}, interval=11.0, resolution=(0.5, 0.25))


def test_build_tiff_metadata_basic_fields():
    ctx = _make_ctx()
    payload = {'fits_io': {'version': 'x', 'axes': 'TZCYX', 'n_channels': 2, 'z_projection': None}}
    out = build_tiff_metadata(ctx, payload)
    assert isinstance(out, TiffMetadata)
    assert out.imagej_meta['axes'] == 'TZCYX'
    assert out.imagej_meta['finterval'] == 11.0
    assert out.imagej_meta['Labels'] == ['GFP', 'mCherry']
    assert out.imagej_meta['unit'] == 'micron'
    assert out.resolution == (2.0, 4.0)


def test_build_tiff_metadata_encodes_payload_in_extratags():
    ctx = _make_ctx()
    payload = {
        'fits_io': {
            'version': 'x',
            'axes': 'TZCYX',
            'n_channels': 2,
            'z_projection': 'mean',
            'source_channel_indices': [0, 1],
            'source_channel_count': 2,
        },
        'project_metadata': {'param': 'value'},
    }
    out = build_tiff_metadata(ctx, payload)
    assert out.extratags is not None
    tag, dtype, count, raw, writeonce = out.extratags[0]
    assert tag == FITS_TAG
    assert dtype == 'B'
    assert count == len(raw)
    assert writeonce is True
    decoded = json.loads(raw.decode('utf-8'))
    assert decoded['fits_io']['z_projection'] == 'mean'
    assert decoded['project_metadata']['param'] == 'value'