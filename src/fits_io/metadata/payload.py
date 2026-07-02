from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

from fits_io.metadata.models import FitsIOPayload
from fits_io.readers._types import Zproj


def _get_fits_io_version() -> str:
    try:
        return version("fits_io")
    except PackageNotFoundError:
        return "unknown"


def build_payload(base: FitsIOPayload, 
                  *, 
                  axes: str | None = None, 
                  channel_labels: Sequence[str] | None = None, 
                  n_channels: int | None = None, 
                  source_channel_indices: Sequence[int] | None = None, 
                  source_channel_count: int | None = None, 
                  z_projection: Zproj = None, 
                  custom_metadata: Mapping[str, Any] | None = None, 
                  compression: str | None = None
                  ) -> FitsIOPayload:
    """
    Build a new FitsIOPayload with the provided metadata (either empty or exising metadata). 
    
    Args:
        base (FitsIOPayload): The base payload to build upon. Can be an empty payload or an existing payload with metadata.
        axes (str): The axes string representing the order of dimensions in the image data.
        channel_labels (Sequence[str]): The list of channel labels.
        n_channels (int): The current number of channels in the image data.
        source_channel_indices (Sequence[int]): The indices of the channels in the source image data.
        source_channel_count (int): The total number of channels in the source image data.
        z_projection (Zproj, optional): The z-projection method used for the image data. Defaults to None.
        custom_metadata (Mapping[str, Any], optional): Additional custom metadata to include in the payload. Defaults to None.
        compression (str, optional): The compression method used for the image data. Defaults to None.
        
    Returns:
        FitsIOPayload: A new payload containing the provided metadata, preserving any existing metadata in the base payload, and adding or updating any custom metadata.
    """
    payload = base.with_fitsio(
        version=_get_fits_io_version(),
        axes=axes,
        channel_labels=channel_labels,
        n_channels=n_channels,
        source_channel_indices=source_channel_indices,
        source_channel_count=source_channel_count,
        z_projection=z_projection,
        compression=compression,)

    if custom_metadata is None:
        custom_metadata = base.custom_metadata
    
    payload = payload.with_custom_metadata(custom_metadata)

    return payload




