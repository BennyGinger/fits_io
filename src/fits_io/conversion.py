from typing import Sequence
import logging

from numpy.typing import NDArray
import numpy as np

from fits_io.image_reader import ImageReader, Zproj


logger = logging.getLogger(__name__)


def resolve_channel_labels(channel_labels: str | Sequence[str] | None, n_channels: int, export_channels: str | Sequence[str]) -> tuple[list[str], bool]:
    """
    Resolve the channel labels to export based on the original channel labels from the reader and the requested export channels. Handles cases where channel labels are not provided, where all channels are requested, and where specific channels are requested but not found in the original labels.
    
    Args:
        channel_labels: The original channel labels from the reader (can be None, a single string, or a sequence of strings).
        n_channels: The number of channels in the image    
        export_channels: The channels requested for export (can be "all", a single string, or a sequence of strings).
    
    Returns:
        A tuple of (resolved_channel_labels, export_all_flag) where resolved_channel_labels is a list of channel labels to export and export_all_flag is a boolean indicating if all channels are being exported
    """
    
    if channel_labels is None:
        out_channel = [f"C_{i+1}" for i in range(n_channels)]
        export_all_flag = True
        logger.debug(f"No channel_labels provided; using default labels {out_channel} and export flag {export_all_flag}")
        return out_channel, export_all_flag

    if isinstance(channel_labels, str):
        labels = [channel_labels]
    else:
        labels = list(channel_labels)
    
    if len(labels) != n_channels:
        raise ValueError(f"Number of channel labels {len(labels)} does not match number of channels {n_channels}.")
    
    if isinstance(export_channels, str) and export_channels.lower() == "all":
        out_channel = labels
        export_all_flag = True
    else:
        requested = [export_channels] if isinstance(export_channels, str) else list(export_channels)
        export_all_flag = False
        
        if any(ch not in labels for ch in requested):
            logger.warning(
                "Requested export channels %s should be in channel labels %s; falling back to all.",
                requested,
                labels,
            )
            out_channel = labels
            export_all_flag = True
        else:
            out_channel = requested
    logger.debug(f"Resolved channel labels for export: {out_channel} from original {channel_labels} with export_all_flag set to {export_all_flag}")
    return out_channel, export_all_flag
        
def get_array_to_export(img_reader: ImageReader, export_channels: list[str], export_all_flag: bool, z_projection: Zproj = None) -> list[NDArray]:
    """
    Get the array(s) to export from the image reader based on the resolved channel labels and export flag. If exporting all channels, retrieves the full array; otherwise, retrieves only the specified channels. Handles cases where the retrieved arrays may be empty and raises an error if so.
    
    Args:
        img_reader: The image reader to retrieve the arrays from.
        export_channels: The resolved channel labels to export.
        export_all_flag: A boolean indicating if all channels are being exported.
        z_projection: An optional Z projection method to apply when retrieving the arrays.
    """
    
    
    if export_all_flag:
        arrays = img_reader.get_array(z_projection)
    else:
        arrays = img_reader.get_channel(export_channels, z_projection)
    
    arrays_list = [arrays] if isinstance(arrays, np.ndarray) else list(arrays)
    if any(a.size == 0 for a in arrays_list):
        raise ValueError("Export produced empty arrays (likely unsupported channel extraction or reader bug).")
    return arrays_list