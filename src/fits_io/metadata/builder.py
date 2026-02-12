from typing import Mapping, Sequence, Any
import logging

from fits_io.readers._types import Zproj, StatusFlag
from fits_io.readers.factory import ImageReader
from fits_io.metadata.models import ChannelMeta, ResolutionMeta, StackMeta, TiffMetadata
from fits_io.metadata.utils import encode_metadata, get_step_name, update_metadata, validate_labels
from fits_io.metadata.provenance import add_provenance_profile


logger = logging.getLogger(__name__)

DEFAULT_DISTRIBUTION = 'unknown_distribution'

def build_metadata(img_reader: ImageReader, *, user_name: str, distribution: str | None = None, step_name: str | None = None, channel_labels: str | Sequence[str] | None = None, z_projection: Zproj = None, extra_step_metadata: Mapping[str, Any] | None = None, add_provenance: bool = True, new_status: StatusFlag | None = None, series_index: int = 0) -> TiffMetadata:
    """
    Build ImageJ-compatible metadata for saving TIFF files.
    
    Args:
        img_reader: An ImageReader instance to read metadata from.
        user_name: Name of the user performing the conversion.
        distribution: Optional; name of the distribution or package.
        step_name: Optional; name of the processing step.
        channel_labels: Optional; either a single string label or a sequence of labels for each channel. If None, default labels will be used.
        z_projection: Optional; method of z-projection applied to the image data.
        extra_step_metadata: Optional mapping of additional metadata to include in the processing step.
        add_provenance: Optional; if True, add provenance information to the metadata.
        new_status: Optional; if provided, overrides the status in the metadata.
        series_index: Optional; index of the series to use for multi-series images, purely to save appropriate metadata.
    
    Returns:
        TiffMetadata object containing metadata, resolution, and extra tags.
    """
    chosen_labels = channel_labels or img_reader.channel_labels
    
    # Determine number of channels for metadata
    if channel_labels is None:
        # exporting all channels
        n_channels = img_reader.channel_number[series_index]
    else:
        if isinstance(channel_labels, str):
            n_channels = 1
        else:
            n_channels = len(list(channel_labels))

    chosen_labels = validate_labels(chosen_labels, n_channels)
    logger.debug(f"Validated channel labels: {chosen_labels}, number of channels: {n_channels}")
    
    # Determine axes string for metadata, adjusting for z-projection and channel export
    axes = img_reader.axes[series_index]
    if z_projection is not None:
        axes = axes.replace('Z', '')  # drop Z axis if z-projection is applied
    if n_channels == 1:
        axes = axes.replace('C', '')  # drop C axis for single channel export
    logger.debug(f"Building metadata with axes: {axes}, channel_labels: {chosen_labels}, z_projection: {z_projection}") 
    
    # build custom metadata to be stored in private tag
    payload = dict(img_reader.custom_metadata)
    
    if add_provenance:
        dist = distribution or DEFAULT_DISTRIBUTION
        step = get_step_name(payload, step_name=step_name)
        payload = add_provenance_profile(payload, distribution=dist, step_name=step)
        payload = update_metadata(payload, update_meta=extra_step_metadata, step_name=step, z_projection=z_projection)
    
    # extract metadata components
    channel_meta = ChannelMeta(channel_number=n_channels, labels=chosen_labels)
    resolution_meta = ResolutionMeta(img_reader.resolution[series_index])
    stack_meta = StackMeta(axes=axes, 
                           status=img_reader.status,
                           user_name=user_name,
                           finterval=img_reader.interval)
    
    if new_status is not None:
        stack_meta.change_status(new_status)
    
    extratags = encode_metadata(payload)
    
    # build final metadata dict
    metadata_dict = stack_meta.to_dict()
    metadata_dict.update(channel_meta.to_dict())
    metadata_dict.update(resolution_meta.to_dict())
    
    return TiffMetadata(
        imagej_meta=metadata_dict,
        resolution=resolution_meta.resolution,
        extratags=extratags)