import numpy as np
import pytest
from pathlib import Path

from fits_io.readers.r_tiff import TiffReader





def test_tiff_axes_channel_series_resolution_interval(monkeypatch, tmp_path: Path, fake_tiff_file_full):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_full)

    r = TiffReader(p)
    assert r.axes == "CYX"
    assert r.channel_count == 3
    assert r.series_count == 2
    assert r.resolution == (0.5, 0.25)
    assert r.interval == 11.0


def test_tiff_get_array_no_series(monkeypatch, tmp_path: Path, fake_tiff_file_no_series_axes):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series_axes)

    r = TiffReader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 5, 6)


def test_tiff_get_array_with_series_splits(monkeypatch, tmp_path: Path, fake_tiff_file_series_S0):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_series_S0)

    r = TiffReader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6)


def test_tiff_zproj_from_fits_io_metadata(tmp_path: Path, fake_tiff_file_with_status, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_status)
    
    r = TiffReader(p)
    assert r.zproj_method == "mean"


def test_tiff_zproj_default_none(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    assert r.zproj_method is None


def test_tiff_metadata_with_payload_json(tmp_path: Path, fake_tiff_file_with_custom_meta, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_custom_meta)

    r = TiffReader(p)
    assert r.metadata.fits_io.z_projection == "max"


def test_tiff_compression_method(tmp_path: Path, fake_tiff_file_with_compression, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_compression)
    
    r = TiffReader(p)
    assert r.compression_method == "ADOBE_DEFLATE"


def test_tiff_channel_number_no_c_axis(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    assert r.channel_count == 1


def test_tiff_channel_number_with_c_axis(tmp_path: Path, fake_tiff_file_with_label_list, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)
    
    r = TiffReader(p)
    assert r.channel_count == 3


def test_tiff_get_array_with_z_projection_max(tmp_path: Path, fake_tiff_file_no_series_axes, monkeypatch):
    """Test z-projection with max method."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    
    # Create array with Z axis shape (2, 3, 4) -> ZYX
    test_arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                         [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]], dtype=np.uint8)
    
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series_axes)
    
    class _FakeSeries:
        axes = "ZYX"
        shape = test_arr.shape
        pages = [type("P", (), {"tags": {"Compression": type("C", (), {"value": 1})()}})()]

        @staticmethod
        def asarray():
            return test_arr

    class _FakeTiff:
        series = [_FakeSeries()]
        imagej_metadata = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", lambda _p: _FakeTiff())
    
    r = TiffReader(p)
    out = r.get_array(z_projection='max')
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)  # Z axis removed
    # Check that max projection worked
    assert out[0, 0] == 2  # max of [1, 2]


def test_tiff_get_channel_single_channel_without_c_axis_returns_full_array(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)

    r = TiffReader(p)
    out = r.get_channel(0)

    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6)


def test_tiff_can_read_valid(tmp_path: Path):
    """Test can_read returns True for valid TIFF extensions."""
    assert TiffReader.can_read(tmp_path / "test.tif")
    assert TiffReader.can_read(tmp_path / "test.tiff")
    assert TiffReader.can_read(tmp_path / "test.TIF")
    assert TiffReader.can_read(tmp_path / "test.TIFF")


def test_tiff_can_read_invalid(tmp_path: Path):
    """Test can_read returns False for invalid extensions."""
    assert not TiffReader.can_read(tmp_path / "test.nd2")
    assert not TiffReader.can_read(tmp_path / "test.png")
    assert not TiffReader.can_read(tmp_path / "test.jpg")



