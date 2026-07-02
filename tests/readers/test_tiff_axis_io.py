from __future__ import annotations

import numpy as np
import pytest

from fits_io.readers.r_tiff import TiffReader


def test_get_channel_single_without_c_axis_returns_full_array(monkeypatch, tmp_path, fake_tiff_file_no_series):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)

    r = TiffReader(p)
    out = r.get_channel(0)

    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6)


def test_get_channel_without_c_axis_nonzero_raises(monkeypatch, tmp_path, fake_tiff_file_no_series):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_no_series)

    r = TiffReader(p)
    with pytest.raises(IndexError, match="out of range"):
        r.get_channel(1)


def test_get_channel_with_c_axis_subset(monkeypatch, tmp_path, fake_tiff_file_with_label_list):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)

    r = TiffReader(p)
    out = r.get_channel([2, 0])

    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 5, 6)


def test_get_channel_with_c_axis_single_drops_c(monkeypatch, tmp_path, fake_tiff_file_with_label_list):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)

    r = TiffReader(p)
    out = r.get_channel(1)

    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 6)


def test_get_channel_empty_selection_raises(monkeypatch, tmp_path, fake_tiff_file_with_label_list):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")
    monkeypatch.setattr("fits_io.readers.r_tiff.TiffFile", fake_tiff_file_with_label_list)

    r = TiffReader(p)
    with pytest.raises(ValueError, match="cannot be empty"):
        r.get_channel([])
