from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import Zproj
from fits_io.readers.protocol import ImageReader
from fits_io.readers.r_tiff import TiffReader
from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.tiff_meta import assemble_tiff_metadata
from fits_io.metadata.resolve import move_or_add_channel_axis, resolve_channel_selection, resolve_merged_axes, resolve_output_axes
from fits_io.metadata.payload import assemble_payload, build_payload
from fits_io.writers.models import ChannelMergeResult, ConversionPreparation, ChannelSelection
from fits_io.writers.core import save_tiff
from fits_io.writers.filesystem import build_output_path



def prepare_conversion(img_reader: ImageReader,
                       *,
                       selection: ChannelSelection,
                       output_name: str,
                       artifact_kind: str | None = None,
                       created_by: str | None = None,
                       custom_metadata: Mapping[str, Any] | None = None,
                       z_projection: Zproj = None,
                       ) -> ConversionPreparation:
    """
    Prepare an array for initial conversion.

    The channel selection must already be resolved so that the same selection
    can subsequently be used to build the output metadata.

    Args:
        img_reader:
            Reader for the source image.
        selection:
            Resolved channels to extract from the source image.
        z_projection:
            Optional z-projection to apply while reading.

    Returns:
        The selected and optionally projected image array.
    """
    array = img_reader.get_channel(channel=selection.export_indices, 
                                   z_projection=z_projection,)
    
    output_path = build_output_path(img_reader, save_name=output_name)
    
    metadata = build_payload(img_reader,
                             selection=selection,
                             artifact_kind=artifact_kind,
                             created_by=created_by,
                             z_projection=z_projection,
                             custom_metadata=custom_metadata,
                             array_shape=array.shape,)
    return ConversionPreparation(array=array, 
                                 output_path=output_path,
                                 metadata=metadata)


def save_array(img_reader: ImageReader, 
               array: NDArray[Any], 
               *, 
               fitsio_metadata: FitsIOMeta,
               output_path: Path, 
               compression: str | None = 'zlib', 
               ) -> Path:
    """
    Save the given array to a FITS TIFF file with ImageJ metadata.
    
    Policy:
    - This function will only save the provided array to a new file with the specified output name.
    - The metadata will be retrieved from the image reader, except for any custom metadata provided, which will be added or merged to existing metadata.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        array : The NumPy array to save.
        output_path : The path of the output TIFF file.
        fitsio_metadata : The FitsIOMeta object containing the metadata to write to the output file.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    
    Returns:
        Path: The path of the saved TIFF file.
    """
    # Check compression method
    if compression is None:
        current_compression = img_reader.compression_method
    else:
        current_compression = compression

    meta_write = assemble_tiff_metadata(fitsio_metadata, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    save_tiff(array, output_path, meta_write, compression=current_compression)
    return output_path


def set_channel_labels(img_reader: ImageReader, 
                       channel_labels: str | Sequence[str], 
                       compression: str | None = 'zlib',
                       ) -> Path:
    """
    Set the channel labels in the metadata.
    
    Policy:
    - This function will change the channel labels in metadata, and re-save it with updated metadata.
    - It will not recreate an artifact, just update the metadata in the existing file.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image. Only reader from .tif is supported.
        channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_channel_labels only supports .tif/.tiff files.")
    
    # Resolve channel labels and export channels
    selection = resolve_channel_selection(channel_labels, 
                                          n_channels=img_reader.channel_count)
    
    # Get existing metadata to preserve other fields
    meta = assemble_payload(img_reader.metadata,
                         channel_labels=selection.export_labels,)
    meta_write = assemble_tiff_metadata(meta, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    # Get array to save (no z-projection applied here)
    array = img_reader.get_array()
    save_tiff(array, img_reader.img_path, meta_write, compression=compression)
    return img_reader.img_path 


def apply_zproj(img_reader: ImageReader, 
                z_projection: Zproj = None, 
                compression: str | None = 'zlib'
                ) -> Path:
    """
    Apply z-projection to the image array and update the file with the projected array and updated metadata.
    
    Policy:
    - This function will apply the specified z-projection to the existing array and re-save it with updated metadata.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image. Only reader from .tif is supported.
        z_projection : The z-projection method to apply ('max', 'mean', or None). If None, no projection is applied.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("apply_zproj only supports .tif/.tiff files.")
    
    out_axes = resolve_output_axes(img_reader.axes, z_projection, img_reader.channel_count)
    
    # Get existing metadata to preserve other fields
    meta = assemble_payload(img_reader.metadata,
                         z_projection=z_projection,
                         axes=out_axes,)
    meta_write = assemble_tiff_metadata(meta, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    # Get array to save
    array = img_reader.get_array(z_projection)
    save_tiff(array, img_reader.img_path, meta_write, compression=compression)
    return img_reader.img_path


def merge_channel_arrays(*,
                         existing_array: NDArray[Any],
                         existing_axes: str,
                         existing_channel_indices: Sequence[int],
                         new_array: NDArray[Any],
                         new_axes: str,
                         new_channel_indices: Sequence[int],
                         reference_axes: str,
                         ) -> ChannelMergeResult:
    """
    Append newly produced channels to an existing channel artifact.

    Arrays without a C axis are treated as single-channel arrays. The output
    channel-axis position follows the existing artifact when it already has a
    C axis; otherwise, it follows the reference axes.
    """
    merged_axes, channel_axis = resolve_merged_axes(
        existing_axes=existing_axes,
        reference_axes=reference_axes,
    )

    existing_with_c = move_or_add_channel_axis(
        array=existing_array,
        axes=existing_axes,
        target_position=channel_axis,
    )
    new_with_c = move_or_add_channel_axis(
        array=new_array,
        axes=new_axes,
        target_position=channel_axis,
    )

    merged_array = np.concatenate(
        [existing_with_c, new_with_c],
        axis=channel_axis,
    )

    return ChannelMergeResult(
        array=merged_array,
        axes=merged_axes,
        channel_indices=[
            *existing_channel_indices,
            *new_channel_indices,
        ],
    )






    

    