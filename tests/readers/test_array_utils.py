"""Tests for array utility functions used in segmentation workflows."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fits_io.readers.array_utils import (
    _axis_index,
    apply_zproj,
    iter_frames,
    render_rgb_like_array,
)


# -------------------------
# Helper function tests
# -------------------------

def test_axis_index_axis_present():
    """Test finding axis that exists in order."""
    assert _axis_index("TCZYX", "T") == 0
    assert _axis_index("TCZYX", "C") == 1
    assert _axis_index("TCZYX", "Y") == 3
    assert _axis_index("TCZYX", "X") == 4


def test_axis_index_axis_absent():
    """Test finding axis that doesn't exist."""
    assert _axis_index("TCZYX", "P") is None
    assert _axis_index("YX", "T") is None


def test_axis_index_case_insensitive():
    """Test that axis search is case insensitive."""
    assert _axis_index("tczyx", "T") == 0
    assert _axis_index("TCZYX", "t") == 0
    assert _axis_index("TcZyX", "z") == 2


def test_apply_zproj_max_projection():
    """Test max projection along Z axis."""
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2) - ZYX
    result = apply_zproj(arr, z_axis=0, zproj="max")
    expected = np.array([[5, 6], [7, 8]])
    assert_array_equal(result, expected)


def test_apply_zproj_mean_projection():
    """Test mean projection along Z axis."""
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    result = apply_zproj(arr, z_axis=0, zproj="mean")
    expected = np.array([[3, 4], [5, 6]], dtype=np.float32)
    assert_array_equal(result, expected)


def test_apply_zproj_invalid_method():
    """Test that invalid projection method raises error."""
    arr = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="Unsupported z-projection"):
        apply_zproj(arr, z_axis=0, zproj="invalid")  # type: ignore


# -------------------------
# iter_frames tests
# -------------------------

def test_iter_frames_over_t_axis():
    """Test iterating over T axis yields correct number of frames."""
    arr = np.arange(12).reshape(3, 2, 2)  # TYX: 3 frames
    frames = list(iter_frames(arr, iterate_axis="T", axis_order="TYX"))
    
    assert len(frames) == 3
    assert frames[0].shape == (2, 2)
    assert_array_equal(frames[0], np.array([[0, 1], [2, 3]]))
    assert_array_equal(frames[1], np.array([[4, 5], [6, 7]]))
    assert_array_equal(frames[2], np.array([[8, 9], [10, 11]]))


def test_iter_frames_axis_not_present():
    """Test that when iterate axis is absent, yields array once."""
    arr = np.arange(4).reshape(2, 2)  # YX only
    frames = list(iter_frames(arr, iterate_axis="T", axis_order="YX"))
    
    assert len(frames) == 1
    assert_array_equal(frames[0], arr)


def test_iter_frames_with_indices():
    """Test iterating with specific indices."""
    arr = np.arange(20).reshape(5, 2, 2)  # TYX: 5 frames
    frames = list(iter_frames(arr, iterate_axis="T", axis_order="TYX", indices=[0, 2, 4]))
    
    assert len(frames) == 3
    assert_array_equal(frames[0], arr[0])
    assert_array_equal(frames[1], arr[2])
    assert_array_equal(frames[2], arr[4])


def test_iter_frames_with_zproj_max():
    """Test iterating with max z-projection."""
    # TZYX: 2 timepoints, 3 z-slices, 2x2 spatial
    arr = np.arange(24).reshape(2, 3, 2, 2)
    frames = list(iter_frames(arr, iterate_axis="T", axis_order="TZYX", zproj="max"))
    
    assert len(frames) == 2
    # After max projection over Z, should have YX shape
    assert frames[0].shape == (2, 2)
    # First timepoint, max over z-slices [0-11]
    assert_array_equal(frames[0], np.array([[8, 9], [10, 11]]))
    # Second timepoint, max over z-slices [12-23]
    assert_array_equal(frames[1], np.array([[20, 21], [22, 23]]))


def test_iter_frames_with_zproj_mean():
    """Test iterating with mean z-projection."""
    # TZYX: 2 timepoints, 3 z-slices, 2x2 spatial
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    frames = list(iter_frames(arr, iterate_axis="T", axis_order="TZYX", zproj="mean"))
    
    assert len(frames) == 2
    assert frames[0].shape == (2, 2)
    # First timepoint, mean over z-slices
    expected_mean_t0 = np.mean(arr[0], axis=0)
    assert_array_equal(frames[0], expected_mean_t0)


def test_iter_frames_z_axis_removed_by_projection():
    """Test that when Z is the only axis and it's projected, yields once."""
    arr = np.arange(6).reshape(3, 2)  # ZX
    frames = list(iter_frames(arr, iterate_axis="Z", axis_order="ZX", zproj="max"))
    
    # After projection, Z is gone, should yield once
    assert len(frames) == 1
    assert_array_equal(frames[0], np.array([4, 5]))  # max over Z


def test_iter_frames_case_insensitive_axis_order():
    """Test that axis_order is case insensitive."""
    arr = np.arange(12).reshape(3, 2, 2)
    frames_upper = list(iter_frames(arr, iterate_axis="T", axis_order="TYX"))
    frames_lower = list(iter_frames(arr, iterate_axis="t", axis_order="tyx"))
    
    assert len(frames_upper) == len(frames_lower)
    for f1, f2 in zip(frames_upper, frames_lower):
        assert_array_equal(f1, f2)


# -------------------------
# render_rgb_like_array tests
# -------------------------

def test_render_rgb_no_channel_axis_adds_one():
    """Test that missing channel axis is added."""
    arr = np.ones((10, 10))  # YX
    result = render_rgb_like_array(arr, axis_order="YX", target_channels=3)
    
    # Should add channel axis and pad to 3 channels
    assert result.shape == (3, 10, 10)  # CYX
    assert_array_equal(result[0], np.ones((10, 10)))
    assert_array_equal(result[1], np.zeros((10, 10)))
    assert_array_equal(result[2], np.zeros((10, 10)))


def test_render_rgb_preserves_dtype():
    """Test that dtype is preserved."""
    arr = np.ones((10, 10), dtype=np.uint8)
    result = render_rgb_like_array(arr, axis_order="YX", target_channels=3)
    
    assert result.dtype == np.uint8


def test_render_rgb_one_channel_pads_to_three():
    """Test padding from 1 channel to 3."""
    arr = np.ones((1, 10, 10))  # CYX with 1 channel
    result = render_rgb_like_array(arr, axis_order="CYX", target_channels=3)
    
    assert result.shape == (3, 10, 10)
    assert_array_equal(result[0], np.ones((10, 10)))
    assert_array_equal(result[1], np.zeros((10, 10)))
    assert_array_equal(result[2], np.zeros((10, 10)))


def test_render_rgb_two_channels_pads_to_three():
    """Test padding from 2 channels to 3."""
    arr = np.stack([np.ones((10, 10)), np.ones((10, 10)) * 2])  # (2, 10, 10)
    result = render_rgb_like_array(arr, axis_order="CYX", target_channels=3)
    
    assert result.shape == (3, 10, 10)
    assert_array_equal(result[0], np.ones((10, 10)))
    assert_array_equal(result[1], np.ones((10, 10)) * 2)
    assert_array_equal(result[2], np.zeros((10, 10)))


def test_render_rgb_three_channels_unchanged():
    """Test that 3 channels remains unchanged."""
    arr = np.stack([np.ones((10, 10)) * i for i in range(3)])  # (3, 10, 10)
    result = render_rgb_like_array(arr, axis_order="CYX", target_channels=3)
    
    assert result.shape == (3, 10, 10)
    assert_array_equal(result, arr)


def test_render_rgb_more_than_three_truncates():
    """Test truncation from >3 channels to 3."""
    arr = np.stack([np.ones((10, 10)) * i for i in range(5)])  # (5, 10, 10)
    result = render_rgb_like_array(arr, axis_order="CYX", target_channels=3)
    
    assert result.shape == (3, 10, 10)
    # Should keep first 3 channels
    assert_array_equal(result[0], np.zeros((10, 10)))
    assert_array_equal(result[1], np.ones((10, 10)))
    assert_array_equal(result[2], np.ones((10, 10)) * 2)


def test_render_rgb_preserves_axis_order():
    """Test that channel axis is returned to original position."""
    # TCZYX: time, channel, z, y, x
    arr = np.ones((2, 1, 3, 10, 10))
    result = render_rgb_like_array(arr, axis_order="TCZYX", target_channels=3)
    
    # Should maintain TCZYX order with C=3
    assert result.shape == (2, 3, 3, 10, 10)
    assert_array_equal(result[:, 0], np.ones((2, 3, 10, 10)))
    assert_array_equal(result[:, 1], np.zeros((2, 3, 10, 10)))
    assert_array_equal(result[:, 2], np.zeros((2, 3, 10, 10)))


def test_render_rgb_channel_at_end():
    """Test when channel axis is at the end."""
    # YXC layout
    arr = np.ones((10, 10, 2))
    result = render_rgb_like_array(arr, axis_order="YXC", target_channels=3)
    
    # Should maintain YXC order with C=3
    assert result.shape == (10, 10, 3)
    assert_array_equal(result[:, :, 0], np.ones((10, 10)))
    assert_array_equal(result[:, :, 1], np.ones((10, 10)))
    assert_array_equal(result[:, :, 2], np.zeros((10, 10)))


def test_render_rgb_case_insensitive_channel_axis():
    """Test that channel_axis parameter is case insensitive."""
    arr = np.ones((1, 10, 10))
    result1 = render_rgb_like_array(arr, axis_order="CYX", channel_axis="C", target_channels=3)
    result2 = render_rgb_like_array(arr, axis_order="CYX", channel_axis="c", target_channels=3)
    
    assert_array_equal(result1, result2)


def test_render_rgb_custom_channel_axis():
    """Test using a custom channel axis letter."""
    # Using P as channel axis
    arr = np.ones((2, 10, 10))
    result = render_rgb_like_array(arr, axis_order="PYX", channel_axis="P", target_channels=3)
    
    assert result.shape == (3, 10, 10)


def test_render_rgb_target_channels_not_3():
    """Test with different target_channels value."""
    arr = np.ones((1, 10, 10))
    result = render_rgb_like_array(arr, axis_order="CYX", target_channels=5)
    
    assert result.shape == (5, 10, 10)
    assert_array_equal(result[0], np.ones((10, 10)))
    for i in range(1, 5):
        assert_array_equal(result[i], np.zeros((10, 10)))
