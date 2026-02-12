from pathlib import Path
import logging
from typing import Sequence


logger = logging.getLogger(__name__)


def _series_has_outputs(save_dir: Path, expected_filenames: set[str]) -> bool:
    """Check if a given series directory contains converted FITS arrays.
    
    Args:
        save_dir: Path to the series directory.
        expected_filenames: Set of expected output filenames for the series, e.g. {"array.tif", "array_zproj.tif"}.
    Returns:
        True if the directory exists and contains at least one file, False otherwise.
    """
    if not save_dir.is_dir():
        return False
    
    return any((save_dir / name).is_file() for name in expected_filenames)

def image_converted(save_dirs: list[Path], expected_filenames: set[str]) -> bool:
    """
    Check if all series of an image have been converted and saved.
    Args:
        save_dirs: List of Paths to the series directories of an image.
        expected_filenames: Set of expected output filenames for each series, e.g. {"array.tif", "array_zproj.tif"}.
        
    Returns:
        True if all series directories exist and contain converted FITS arrays, False otherwise.
    """
    return all(_series_has_outputs(d, expected_filenames) for d in save_dirs)

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