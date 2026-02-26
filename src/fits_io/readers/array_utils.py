from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import ArrAxis, Zproj


def _axis_index(axis_order: str, axis: str) -> int | None:
    """
    Return the index of a given axis in the axis order string.
    
    Args:
        axis_order: String describing axis order (e.g., "TCZYX").
        axis: Single character representing the axis to find.
    
    Returns:
        Index of the axis, or None if not present.
    """
    axis_upper = axis.upper()
    order_upper = axis_order.upper()
    
    try:
        return order_upper.index(axis_upper)
    except ValueError:
        return None


def apply_zproj(arr: NDArray, z_axis: int | None, zproj: Zproj | None) -> NDArray:
    """
    Apply z-projection to an array along the specified axis.
    
    Args:
        arr: Input array.
        z_axis: Axis index for Z dimension.
        zproj: Projection method ('max' or 'mean').
    
    Returns:
        Projected array with Z dimension removed.
    """
    if z_axis is None or zproj is None:
        return arr
    
    if zproj == 'max':
        return np.max(arr, axis=z_axis)
    elif zproj == 'mean':
        return np.mean(arr, axis=z_axis)
    else:
        raise ValueError(f"Unsupported z-projection method: {zproj}")


def iter_frames(arr: NDArray, *, iterate_axis: ArrAxis = "T", axis_order: str = "TYX", indices: Sequence[int] |  None = None, zproj: Zproj = None) -> Iterator[NDArray]:
    """
    Iterate over frames along a specified axis, with optional z-projection.
    
    Args:
        arr: Input array to iterate over.
        iterate_axis: Single character representing the axis to iterate over (default "T").
        axis_order: String describing the current axis order of arr (e.g., "TCZYX").
        indices: Optional sequence of indices to yield from the iteration axis.
        zproj: Optional z-projection method ('max' or 'mean'). Applied before iteration if Z axis exists.
    
    Yields:
        NDArray frames along the specified axis.
    
    Notes:
        - If iterate_axis is not in axis_order, yields arr once and returns.
        - If Z exists in axis_order and zproj is provided, applies projection first.
        - Yields frames according to indices if provided, otherwise all frames.
    """
    # Check if iterate axis exists in the axis order
    iterate_idx = _axis_index(axis_order, iterate_axis)
    
    if iterate_idx is None:
        # Axis not present, yield the array once
        yield arr
        return
    
    # Apply z-projection if requested
    working_arr = arr
    working_axis_order = axis_order.upper()
    
    if zproj is not None:
        z_idx = _axis_index(axis_order, "Z")
        if z_idx is not None:
            working_arr = apply_zproj(working_arr, z_idx, zproj)
            # Remove Z from axis order and recompute iterate axis index
            working_axis_order = working_axis_order.replace("Z", "")
            iterate_idx = _axis_index(working_axis_order, iterate_axis)
            
            if iterate_idx is None:
                # After removing Z, iterate axis no longer exists
                yield working_arr
                return
    
    # Move iterate axis to the front
    working_arr = np.moveaxis(working_arr, iterate_idx, 0)
    
    # Determine which indices to yield
    if indices is not None:
        selected_indices = indices
    else:
        selected_indices = range(working_arr.shape[0])
    
    # Yield frames
    for idx in selected_indices:
        yield working_arr[idx]


def render_rgb_like_array(arr: NDArray, *, axis_order: str, channel_axis: ArrAxis = "C", target_channels: int = 3) -> NDArray:
    """
    Ensure an array has exactly target_channels channels.
    
    Args:
        arr: Input array.
        axis_order: String describing the current axis order of arr (e.g., "CZYX").
        channel_axis: Single character representing the channel axis (default "C").
        target_channels: Desired number of channels (default 3).
    
    Returns:
        Array with exactly target_channels channels, preserving original axis layout.
    
    Notes:
        - If channel_axis not in axis_order, adds a new channel axis at the front.
        - If nC > target_channels, truncates to first target_channels.
        - If nC < target_channels, pads with zeros to reach target_channels.
        - Output preserves the original axis order and dtype.
    """
    working_arr = arr
    working_axis_order = axis_order.upper()
    channel_idx = _axis_index(working_axis_order, channel_axis)
    channel_upper = channel_axis.upper()
    
    # Check if channel axis exists
    
    if channel_idx is None:
        # Add a new channel axis at the front
        working_arr = np.expand_dims(working_arr, axis=0)
        working_axis_order = channel_upper + working_axis_order
        channel_idx = 0
    
    # Move channel axis to position 0 for processing
    original_channel_idx = channel_idx
    working_arr = np.moveaxis(working_arr, channel_idx, 0)
    
    nC = working_arr.shape[0]
    
    # Adjust number of channels
    if nC == target_channels:
        # Already correct number of channels
        result = working_arr
    elif nC > target_channels:
        # Truncate to target_channels
        result = working_arr[:target_channels]
    else:
        # Pad with zeros to reach target_channels
        padding_shape = (target_channels - nC,) + working_arr.shape[1:]
        padding = np.zeros(padding_shape, dtype=working_arr.dtype)
        result = np.concatenate([working_arr, padding], axis=0)
    
    # Move channel axis back to original position
    result = np.moveaxis(result, 0, original_channel_idx)
    
    return result
