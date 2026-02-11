import numpy as np
import pytest
from pathlib import Path

import fits_io.image_reader as m


def test_get_reader_tif(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m, "TiffFile", fake_tiff_file_no_series)

    r = m.get_reader(p)
    assert isinstance(r, m.TiffReader)


def test_get_reader_nd2(tmp_path: Path, monkeypatch, fake_nd2_file_basic):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", fake_nd2_file_basic)

    r = m.get_reader(p)
    assert isinstance(r, m.Nd2Reader)


def test_tiff_axes_channel_series_resolution_interval(monkeypatch, tmp_path: Path, fake_tiff_file_full):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m, "TiffFile", fake_tiff_file_full)

    r = m.TiffReader(p)
    assert r.axes == ["CYX", "CYX"]
    assert r.channel_number == [3, 3]
    assert r.series_number == 2
    assert r.resolution == [(0.5, 0.25), (0.5, 0.25)]
    assert r.interval == 11.0


def test_tiff_get_array_no_series(monkeypatch, tmp_path: Path, fake_tiff_file_no_series_axes):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m, "TiffFile", fake_tiff_file_no_series_axes)
    monkeypatch.setattr(m, "imread", lambda _: np.zeros((2, 3, 4), dtype=np.uint8))

    r = m.TiffReader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3, 4)


def test_tiff_get_array_with_series_splits(monkeypatch, tmp_path: Path, fake_tiff_file_series_S0):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m, "TiffFile", fake_tiff_file_series_S0)
    monkeypatch.setattr(m, "imread", lambda _: np.zeros((2, 5, 6), dtype=np.uint8))  # S,Y,X

    r = m.TiffReader(p)
    out = r.get_array()
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (5, 6)


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

    monkeypatch.setattr(m.nd2, "ND2File", fake_nd2_file_meta(nd2_meta_factories[meta_key]))

    r = m.Nd2Reader(p)
    assert r.resolution == [expected]


def test_nd2_interval_timeloop(monkeypatch, tmp_path: Path, fake_nd2_file_timeloop):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", fake_nd2_file_timeloop)

    r = m.Nd2Reader(p)
    assert r.interval == 5


def test_nd2_interval_netimeloop(monkeypatch, tmp_path: Path, fake_nd2_file_netimeloop):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", fake_nd2_file_netimeloop)

    r = m.Nd2Reader(p)
    assert r.interval == 7


def test_nd2_get_array_no_series(monkeypatch, tmp_path: Path, fake_nd2_file_noP):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", fake_nd2_file_noP)
    monkeypatch.setattr(m.nd2, "imread", lambda _: np.zeros((3, 4), dtype=np.uint16))

    r = m.Nd2Reader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)
