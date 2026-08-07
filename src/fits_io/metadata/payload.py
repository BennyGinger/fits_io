from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.resolve import remap_channel_indices, resolve_output_axes
from fits_io.readers._types import Zproj
from fits_io.readers.protocol import ImageReader
from fits_io.writers.models import ChannelSelection



def _get_dist_version(distribution: str | None) -> str:
    dist = distribution or "fits_io"
    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def build_payload(reader: ImageReader,
                  *,
                  selection: ChannelSelection,
                  artifact_kind: str | None = None,
                  created_by: str | None = None,
                  z_projection: Zproj = None,
                  custom_metadata: Mapping[str, Any] | None = None,
                  array_shape: tuple[int, ...] | None = None,
                  ) -> FitsIOMeta:
    """
    Build a new FitsIOMeta payload based on the provided reader and metadata parameters. This function resolves channel selections, output axes, and remaps source indices as needed.
    
    Args:
        reader (ImageReader): 
            The image reader containing the base metadata.
        selection (ChannelSelection): 
            The resolved channel selection for the output artifact.
        artifact_kind (str | None): 
            The type of artifact being processed (e.g., "raw_image", "segmentation"). If None, the artifact type from the reader's metadata will be used.
        created_by (str | None): 
            The name of the software or tool that created the artifact. If None, the created_by from the reader's metadata will be used.
        z_projection (Zproj): 
            The z-projection method to apply ('max', 'mean', or None).
        custom_metadata (Mapping[str, Any] | None): 
            Additional custom metadata to include in the payload. If None, the custom metadata from the reader's metadata will be used.
        array_shape (tuple[int, ...] | None): 
            The shape of the image array. If provided, it will be used to validate the channel selection against the expected output axes.
    
    Returns:
        FitsIOMeta: A new payload containing the provided metadata, preserving any existing metadata in the reader's metadata, and adding or updating any custom metadata.
    """
    artifact_meta = reader.metadata.fits_io
    
    current_zproj = z_projection
    if current_zproj is None and artifact_meta is not None:
        current_zproj = artifact_meta.z_projection

    out_axes = resolve_output_axes(reader.axes,
                                   current_zproj,
                                   len(selection.export_labels),)

    if array_shape is not None:
        selection.validate_array(array_shape, out_axes)

    # Raw inputs do not necessarily carry channel lineage metadata yet.
    source_channel_indices = (artifact_meta.source_channel_indices
                              or list(range(reader.channel_count)))
    current_artifact_indices = (artifact_meta.artifact_channel_indices
                                or source_channel_indices)
    
    remapped_indices = remap_channel_indices(
        current_artifact_indices=current_artifact_indices,
        selected_local_indices=selection.export_indices,)

    resolved_artifact_type = artifact_kind
    if resolved_artifact_type is None and artifact_meta is not None:
        resolved_artifact_type = artifact_meta.artifact_type

    resolved_created_by = created_by
    if resolved_created_by is None and artifact_meta is not None:
        resolved_created_by = artifact_meta.created_by

    resolved_derived_from = artifact_meta.artifact_type

    return assemble_payload(
        reader.metadata,
        axes=out_axes,
        channel_labels=selection.export_labels,
        artifact_type=resolved_artifact_type,
        created_by=resolved_created_by,
        derived_from=resolved_derived_from,
        source_channel_indices=source_channel_indices,
        artifact_channel_indices=remapped_indices,
        z_projection=current_zproj,
        custom_metadata=custom_metadata,)
    

def assemble_payload(base: FitsIOMeta, 
                  *, 
                  axes: str | None = None, 
                  channel_labels: Sequence[str] | None = None,
                  artifact_type: str | None = None,
                  created_by: str | None = None,
                  derived_from: str | None = None,
                  source_channel_indices: Sequence[int] | None = None, 
                  artifact_channel_indices: Sequence[int] | None = None, 
                  z_projection: Zproj = None, 
                  custom_metadata: Mapping[str, Any] | None = None, 
                  ) -> FitsIOMeta:
    """
    Build a new FitsIOPayload with the provided metadata (either empty or exising metadata). 
    
    Args:
        base (FitsIOPayload): 
            The base payload to build upon.
        axes (str): 
            The axes string representing the order of dimensions in the image data.
        channel_labels (Sequence[str]): 
            The list of channel labels.
        artifact_type (str): 
            The type of artifact being processed (e.g., "raw_image", "segmentation").
        created_by (str): 
            The name of the software or tool that created the artifact.
        derived_from (str): 
            The type of artifact from which this artifact was derived.
        source_channel_indices (Sequence[int]): 
            The indices of the channels in the source image data.
        artifact_channel_indices (Sequence[int]): 
            The indices of the channels in this artifact image data.
        z_projection (Zproj, optional): 
            The z-projection method used for the image data. Defaults to None.
        custom_metadata (Mapping[str, Any], optional): 
            Additional custom metadata to include in the payload. Defaults to None.
        
    Returns:
        FitsIOPayload: A new payload containing the provided metadata, preserving any existing metadata in the base payload, and adding or updating any custom metadata.
    """
    payload = base.with_fitsio(
        artifact_type=artifact_type,
        created_by=created_by,
        version=_get_dist_version(created_by),
        derived_from=derived_from,
        axes=axes,
        channel_labels=channel_labels,
        source_channel_indices=source_channel_indices,
        artifact_channel_indices=artifact_channel_indices,
        z_projection=z_projection)

    if custom_metadata is None:
        custom_metadata = base.custom_metadata
    
    payload = payload.with_custom_metadata(custom_metadata)

    return payload




