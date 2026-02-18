from pathlib import Path
import pytest
import numpy as np
import nd2

from fits_io.readers.r_nd2 import Nd2Reader


@pytest.mark.parametrize(
    "meta_key, expected",
    [
        ("no_channels", None),
        ("no_volume", None),
        ("no_calib", None),
        ("calib", (0.3223, 0.3223)),
    ],
)
def test_nd2_resolution_defensive(monkeypatch, tmp_path: Path, nd2_meta_factories, fake_nd2_file_meta, meta_key, expected):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")

    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_meta(nd2_meta_factories[meta_key]))

    r = Nd2Reader(p)
    assert r.resolution == [expected]


def test_nd2_interval_timeloop(monkeypatch, tmp_path: Path, fake_nd2_file_timeloop):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_timeloop)

    r = Nd2Reader(p)
    assert r.interval == 5


def test_nd2_interval_netimeloop(monkeypatch, tmp_path: Path, fake_nd2_file_netimeloop):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_netimeloop)

    r = Nd2Reader(p)
    assert r.interval == 7


def test_nd2_get_array_no_series(monkeypatch, tmp_path: Path, fake_nd2_file_noP):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_noP)
    monkeypatch.setattr(nd2, "imread", lambda _: np.zeros((3, 4), dtype=np.uint16))

    r = Nd2Reader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)
    
def test_nd2_normalize_channels_valid_int(monkeypatch, tmp_path, fake_nd2_file_3channels):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_3channels)
    
    r = Nd2Reader(p)
    assert r._normalize_channels(1) == [1]
    assert r._normalize_channels([0, 2]) == [0, 2]

def test_nd2_normalize_channels_invalid_int(monkeypatch, tmp_path, fake_nd2_file_3channels):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_3channels)
    
    r = Nd2Reader(p)
    with pytest.raises(IndexError, match="out of range"):
        r._normalize_channels(5)

def test_nd2_normalize_channels_by_label(monkeypatch, tmp_path, fake_nd2_file_with_labels):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_with_labels)
    
    r = Nd2Reader(p, _channel_labels=["DAPI", "GFP", "RFP"])
    assert r._normalize_channels("GFP") == [1]
    assert r._normalize_channels(["DAPI", "RFP"]) == [0, 2]

def test_nd2_normalize_channels_label_not_supported(monkeypatch, tmp_path, fake_nd2_file_no_native_labels):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_no_native_labels)
    
    r = Nd2Reader(p)  # no channel_labels provided
    with pytest.raises(ValueError, match="does not support native channel labels"):
        r._normalize_channels("DAPI")