from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import logging

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import Zproj, validate_axes
from fits_io.writers.models import ChannelSelection


logger = logging.getLogger(__name__)


############# Public APIs ##############################

def resolve_channel_selection(channel_labels: str | Sequence[str] | None, 
                              n_channels: int, 
                              export_channels: str | Sequence[str] | None = None ) -> ChannelSelection:
    """
    Resolve the channel selection for export based on the original channel labels and the requested export channels.
    
    Args:
        channel_labels: The original channel labels from the reader (can be None, a single string, or a sequence of strings).
        n_channels: The number of channels in the image.
        export_channels (Optional): The requested export channels (can be "all", a single string, or a sequence of strings). If None, all channels will be exported.
    
    Returns:
        A ChannelSelection object containing the source labels, export labels, and export indices.
    """
    source_labels = _resolve_channel_labels(channel_labels, n_channels)
    export_labels, export_indices = _resolve_export_channels(export_channels, source_labels)
    return ChannelSelection(source_labels=source_labels, 
                            export_labels=export_labels, 
                            export_indices=export_indices)


def resolve_output_axes(reader_axes: str, z_projection: Zproj, n_channels: int) -> str:
    """
    Resolve output axes string for ImageJ metadata.
    """
    axes = reader_axes
    validate_axes(axes)
    
    if z_projection is not None:
        axes = axes.replace('Z', '')
    if n_channels == 1:
        axes = axes.replace('C', '')
    return axes


def remap_channel_indices(current_artifact_indices: list[int] | None, selected_local_indices: list[int]) -> list[int]:
    """
    Remap the selected indices to the existing indices if provided. If existing is None, return
    the selected indices as is.
    """
    if current_artifact_indices is None:
        return selected_local_indices
    return [current_artifact_indices[i] for i in selected_local_indices]


def resolve_merged_axes(*, existing_axes: str, reference_axes: str,) -> tuple[str, int]:
    """
    Resolve the merged axes and position of the channel axis.
    """
    if "C" in existing_axes:
        return existing_axes, existing_axes.index("C")

    if "C" in reference_axes:
        channel_position = reference_axes.index("C")
    else:
        try:
            channel_position = existing_axes.index("Y")
        except ValueError as exc:
            raise ValueError(f"Cannot insert a channel axis into axes {existing_axes!r}: "
                             "the reference has no C axis and the existing axes have no Y axis."
                             ) from exc

    merged_axes = (existing_axes[:channel_position]+ "C" + existing_axes[channel_position:])

    return merged_axes, channel_position


def move_or_add_channel_axis(*,
                              array: NDArray[Any],
                              axes: str,
                              target_position: int,
                              ) -> NDArray[Any]:
    """
    Ensure that an array has its channel axis at the requested position.
    """
    if "C" not in axes:
        return np.expand_dims(array, axis=target_position)

    current_position = axes.index("C")
    if current_position == target_position:
        return array

    return np.moveaxis(array, current_position, target_position)
############ Private helper functions ###########################

def _resolve_channel_labels(channel_labels: str | Sequence[str] | None, n_channels: int) -> list[str]:
    if channel_labels is None:
        return [f"C_{i+1}" for i in range(n_channels)]
    
    if isinstance(channel_labels, str):
        labels = [channel_labels]
    else:
        labels = list(channel_labels)
        
    for lbl in labels:
        if not isinstance(lbl, str):
            raise ValueError(f"Channel label {lbl} is not a string.")
    
    _ensure_unique(labels, name="channel_labels")
    
    if len(labels) != n_channels:
        logger.warning(f"Number of channel labels {len(labels)} does not match number of channels {n_channels}. Revert to default labels.")
        return [f"C_{i+1}" for i in range(n_channels)]
    
    return labels
    

def _resolve_export_channels(export_channels: str | Sequence[str] | None, channel_labels: list[str]) -> tuple[list[str], list[int]]:
    if export_channels is None:
        return channel_labels, list(range(len(channel_labels)))
    
    if isinstance(export_channels, str):
        if export_channels.lower() == "all":
            return channel_labels, list(range(len(channel_labels)))
        else:
            export_channels = [export_channels]
    
    if len(export_channels) > len(channel_labels):
        raise ValueError(f"Requested export channels {export_channels} exceed available channel labels {channel_labels}.")
    _ensure_unique(export_channels, name="export_channels")
    
    indices: list[int] = []
    for ch in export_channels:
        if ch not in channel_labels:
            raise ValueError(f"Requested export channel '{ch}' not found in channel labels {channel_labels}.")
        indices.append(channel_labels.index(ch))   
    return list(export_channels), indices


def _ensure_unique(values: Sequence[str], *, name: str = "values") -> None:
    seen: set[str] = set()

    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {name}: {value!r}")
        seen.add(value)

