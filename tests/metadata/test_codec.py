import json

from fits_io.metadata.codec import decode_metadata, encode_metadata
from fits_io.metadata.provenance import FITS_TAG


def test_encode_metadata_with_data_returns_tiff_tag_payload():
    payload = {'key': 'value', 'number': 42}
    result = encode_metadata(payload)
    assert result is not None
    tag, dtype, count, raw, writeonce = result[0]
    assert tag == FITS_TAG
    assert dtype == 'B'
    assert count == len(raw)
    assert writeonce is True
    assert json.loads(raw.decode('utf-8')) == payload


def test_encode_metadata_empty_returns_none():
    assert encode_metadata({}) is None


def test_decode_metadata_accepts_bytes():
    raw = b'{"a": 1, "b": "x"}'
    assert decode_metadata(raw) == {'a': 1, 'b': 'x'}


def test_decode_metadata_accepts_str():
    raw = '{"a": 1, "b": "x"}'
    assert decode_metadata(raw) == {'a': 1, 'b': 'x'}