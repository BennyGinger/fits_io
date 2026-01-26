from __future__ import annotations

import numpy as np
import pytest
import tifffile as tiff

from fits_io.reader_utility import read_tiff_channels


@pytest.fixture()
def tiff_tzcyx(tmp_path):
    """
    Create a small TZCYX ImageJ TIFF with Labels for channel names.
    """
    path = tmp_path / "tzcyx.tif"
    rng = np.random.default_rng(0)

    # Keep this small to make tests fast
    T, Z, C, Y, X = 2, 3, 3, 16, 16
    data = rng.integers(0, 65535, size=(T, Z, C, Y, X), dtype=np.uint16)
    labels = ["GFP", "RFP", "DAPI"]

    tiff.imwrite(
        path,
        data,
        imagej=True,
        metadata={"axes": "TZCYX", "Labels": labels},
    )
    return path, data, labels


def test_read_single_channel_by_index_drops_c(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    out = read_tiff_channels(path, 1, channel_labels=labels)
    assert out.shape == data[:, :, 1, :, :].shape  # (T, Z, Y, X)
    np.testing.assert_array_equal(out, data[:, :, 1, :, :])


def test_read_single_channel_by_label_drops_c(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    out = read_tiff_channels(path, "RFP", channel_labels=labels)
    assert out.shape == data[:, :, 1, :, :].shape
    np.testing.assert_array_equal(out, data[:, :, 1, :, :])


def test_read_multiple_channels_preserves_c_and_order(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    out = read_tiff_channels(path, [2, 0], channel_labels=labels)
    assert out.shape == (data.shape[0], data.shape[1], 2, data.shape[3], data.shape[4])  # (T,Z,Csel,Y,X)
    np.testing.assert_array_equal(out[:, :, 0, :, :], data[:, :, 2, :, :])
    np.testing.assert_array_equal(out[:, :, 1, :, :], data[:, :, 0, :, :])


def test_read_multiple_channels_by_label(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    out = read_tiff_channels(path, ["GFP", "DAPI"], channel_labels=labels)
    np.testing.assert_array_equal(out[:, :, 0, :, :], data[:, :, 0, :, :])
    np.testing.assert_array_equal(out[:, :, 1, :, :], data[:, :, 2, :, :])


def test_duplicates_are_dropped_but_order_preserved(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    out = read_tiff_channels(path, [1, 1, 0, 1], channel_labels=labels)
    # duplicates dropped -> [1, 0]
    assert out.shape[2] == 2
    np.testing.assert_array_equal(out[:, :, 0, :, :], data[:, :, 1, :, :])
    np.testing.assert_array_equal(out[:, :, 1, :, :], data[:, :, 0, :, :])


def test_empty_selection_raises(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    with pytest.raises(ValueError, match="cannot be empty"):
        read_tiff_channels(path, [], channel_labels=labels)


def test_mix_int_and_str_raises(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    with pytest.raises(TypeError, match="mix"):
        read_tiff_channels(path, [0, "RFP"], channel_labels=labels)


def test_label_without_channel_labels_raises(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    with pytest.raises(ValueError, match="channel_labels"):
        read_tiff_channels(path, "RFP", channel_labels=None)


def test_out_of_range_channel_raises(tiff_tzcyx):
    path, data, labels = tiff_tzcyx

    with pytest.raises(IndexError):
        read_tiff_channels(path, 99, channel_labels=labels)
