from dataclasses import dataclass
from typing import Sequence
import logging

from fits_io.readers._types import Zproj, validate_axes


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ChannelSelection:
    """
    Represents the selection of channels for export, including the source labels, export labels, and export indices.
    
    Attributes:
        source_labels: The original channel labels from the reader.
        export_labels: The labels of the channels to be exported.
        export_indices: The indices of the channels to be exported, corresponding to the source labels.
    """
    source_labels: list[str]
    export_labels: list[str]
    export_indices: list[int]
    
    def validate_array(self, array_shape: tuple[int, ...], axes: str) -> None:
        """
        Validate that the provided array shape and axes match the expected channel selection.
        
        Args:
            array_shape: The shape of the array to validate.
            axes: The axes string corresponding to the array.
        """
        validate_axes(axes)

        if len(array_shape) != len(axes):
            raise ValueError(f"Array shape {array_shape} does not match axes '{axes}'.")

        c_index = axes.find('C')
        if c_index == -1:
            if self.export_indices != [0]:
                raise ValueError(f"Array has no channel axis, but export indices are {self.export_indices}.")
            else:
                return  # No channel axis and only one channel selected, valid case.
        c_axis_size = array_shape[c_index]
        if c_axis_size != len(self.export_indices):
            raise ValueError(f"Array channel axis size {c_axis_size} does not match number of export channels {len(self.export_indices)}.")


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
    return ChannelSelection(source_labels=source_labels, export_labels=export_labels, export_indices=export_indices)


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


def remap_source_indices(existing: list[int] | None, selected: list[int]) -> list[int]:
    """
    Remap the selected indices to the existing indices if provided. If existing is None, return
    the selected indices as is.
    """
    if existing is None:
        return selected
    return [existing[i] for i in selected]


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

