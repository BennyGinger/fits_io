from __future__ import annotations

import numpy as np
import pytest
import tifffile

from fits_io.readers.r_tiff import TiffReader


@pytest.mark.parametrize("axes,shape", [
    ("TCZYX", (2, 3, 4, 5, 6)),
    ("TZCYX", (2, 4, 3, 5, 6)),
    ("ZYX", (4, 5, 6)),
])
@pytest.mark.parametrize("channels", [[0], [2, 0]])
def test_channel_projection_uses_axes_after_channel_selection(tmp_path, axes, shape, channels):
    array = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    path = tmp_path / "zstack.tif"
    tifffile.imwrite(path, array, photometric="minisblack", metadata={"axes": axes})
    reader = TiffReader(path)
    np.testing.assert_array_equal(reader.get_array(z_projection="max"),
                                  array.max(axis=axes.index("Z")))

    if "C" in axes:
        selected = channels[0] if len(channels) == 1 else channels
        expected = np.take(array, selected, axis=axes.index("C"))
        output_axes = axes.replace("C", "") if len(channels) == 1 else axes
    else:
        channels = [0]
        expected = array
        output_axes = axes
    expected = expected.max(axis=output_axes.index("Z"))
    np.testing.assert_array_equal(reader.get_channel(channels, z_projection="max"), expected)


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
