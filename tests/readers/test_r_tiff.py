import numpy as np
import pytest
from pathlib import Path
from tifffile import TiffFile, imread

from fits_io.readers.r_tiff import TiffReader





def test_tiff_axes_channel_series_resolution_interval(monkeypatch, tmp_path: Path, fake_tiff_file_full):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_full)

    r = TiffReader(p)
    assert r.axes == ["CYX", "CYX"]
    assert r.channel_number == [3, 3]
    assert r.series_number == 2
    assert r.resolution == [(0.5, 0.25), (0.5, 0.25)]
    assert r.interval == 11.0


def test_tiff_get_array_no_series(monkeypatch, tmp_path: Path, fake_tiff_file_no_series_axes):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series_axes)
    monkeypatch.setattr("fits_io.readers.r_tiff.imread", lambda _: np.zeros((2, 3, 4), dtype=np.uint8))

    r = TiffReader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3, 4)


def test_tiff_get_array_with_series_splits(monkeypatch, tmp_path: Path, fake_tiff_file_series_S0):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_series_S0)
    monkeypatch.setattr("fits_io.readers.r_tiff.imread", lambda _: np.zeros((2, 5, 6), dtype=np.uint8))  # S,Y,X

    r = TiffReader(p)
    out = r.get_array()
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (5, 6)


def test_tiff_parse_info_valid(tmp_path: Path, fake_tiff_file_with_info, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_info)
    
    r = TiffReader(p)
    info = r._parse_info()
    assert info == {"key1": "value1", "key2": "value2"}


def test_tiff_parse_info_no_info(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    info = r._parse_info()
    assert info == {}


def test_tiff_status_from_custom_metadata(tmp_path: Path, fake_tiff_file_with_status, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_status)
    
    r = TiffReader(p)
    assert r.status == "skip"


def test_tiff_status_default(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    assert r.status == "active"


def test_tiff_channel_labels_string(tmp_path: Path, fake_tiff_file_with_single_label, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_single_label)
    
    r = TiffReader(p)
    assert r.channel_labels == ["DAPI"]


def test_tiff_channel_labels_list(tmp_path: Path, fake_tiff_file_with_label_list, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)
    
    r = TiffReader(p)
    assert r.channel_labels == ["DAPI", "GFP", "RFP"]


def test_tiff_channel_labels_none(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    assert r.channel_labels is None


def test_tiff_custom_metadata_valid_json(tmp_path: Path, fake_tiff_file_with_custom_meta, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_custom_meta)
    
    r = TiffReader(p)
    assert r.custom_metadata == {"status": "active", "extra": "data"}


def test_tiff_custom_metadata_no_tag(tmp_path: Path, fake_tiff_file_no_series, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)
    
    r = TiffReader(p)
    assert r.custom_metadata == {}


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
    assert r.channel_number == [1]


def test_tiff_channel_number_with_c_axis(tmp_path: Path, fake_tiff_file_with_label_list, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)
    
    r = TiffReader(p)
    assert r.channel_number == [3]


def test_tiff_get_array_with_z_projection_max(tmp_path: Path, fake_tiff_file_no_series_axes, monkeypatch):
    """Test z-projection with max method."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    
    # Create array with Z axis shape (2, 3, 4) -> ZYX
    test_arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                         [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]], dtype=np.uint8)
    
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series_axes)
    monkeypatch.setattr("fits_io.readers.r_tiff.imread", lambda _: test_arr)
    
    r = TiffReader(p)
    out = r.get_array(z_projection='max')
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)  # Z axis removed
    # Check that max projection worked
    assert out[0, 0] == 2  # max of [1, 2]


def test_tiff_axis_index(tmp_path: Path, fake_tiff_file_full, monkeypatch):
    """Test axis_index returns correct indices for each series."""
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_full)
    
    r = TiffReader(p)
    assert r.axis_index('C') == [0, 0]  # C is at index 0 in both series
    assert r.axis_index('Y') == [1, 1]  # Y is at index 1 in both series
    assert r.axis_index('X') == [2, 2]  # X is at index 2 in both series
    assert r.axis_index('Z') == [None, None]  # Z not present
    assert r.axis_index('T') == [None, None]  # T not present


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



