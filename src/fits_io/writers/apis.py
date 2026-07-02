from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Any

from numpy.typing import NDArray

from fits_io.readers._types import Zproj
from fits_io.readers.protocol import ImageReader
from fits_io.readers.r_tiff import TiffReader
from fits_io.metadata.tiff_meta import assemble_tiff_metadata
from fits_io.metadata.resolve import resolve_channel_selection, resolve_output_axes
from fits_io.metadata.payload import build_payload
from fits_io.writers.filesystem import build_output_paths
from fits_io.writers.core import save_tiff

DEFAULT_OUTPUT_NAME = 'fits.tif'


def convert_to_fits_tif(img_reader: ImageReader, 
                        *, 
                        output_name: str = DEFAULT_OUTPUT_NAME, 
                        channel_labels: str | Sequence[str] | None = None, 
                        export_channels: str | Sequence[str] = 'all', 
                        custom_metadata: Mapping[str, Any] | None = None, 
                        z_projection: Zproj = None, 
                        compression: str | None = 'zlib'
                        ) -> list[Path]:
    """
    Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_reader : An ImageReader instance for the input image.
        output_name : Optional name of the output TIFF file.
        channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
        export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'.
        custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
        z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    Returns:
        List of Paths of the saved TIFF files.
    """
    # Split into series if applicable
    series_readers = img_reader.split_series()
    
    # Get the save directories of the image and generate the save path(s)
    save_path_lst = build_output_paths(series_readers, output_name)
    
    # Resolve channel labels and export channels
    selection = resolve_channel_selection(channel_labels, 
                                          n_channels=img_reader.channel_count,
                                          export_channels=export_channels)

    # Get the image array(s)
    arrays = [reader.get_channel(selection.export_indices, z_projection)
                                  for reader in series_readers]
    
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    # Build metadata and save each array with its corresponding path
    for array, path, reader in zip(arrays, save_path_lst, series_readers):
        out_axes = resolve_output_axes(reader_axes=reader.axes, 
                                       z_projection=z_projection, 
                                       n_channels=len(selection.export_labels))
        meta = build_payload(reader.metadata,
                             axes = out_axes,
                             channel_labels = selection.export_labels,
                             n_channels = len(selection.export_labels),
                             source_channel_indices = selection.export_indices,
                             source_channel_count = reader.channel_count,
                             z_projection = z_projection,
                             custom_metadata = custom_metadata,
                             compression = compression,)
        meta_write = assemble_tiff_metadata(meta, 
                                            reader.interval, 
                                            reader.resolution)
        # save TIFF
        save_tiff(array, path, meta_write, compression=compression)
    return save_path_lst


def save_array(img_reader: ImageReader, 
               array: NDArray[Any], 
               output_name: str = DEFAULT_OUTPUT_NAME, 
               *, 
               custom_metadata: Mapping[str, Any] | None = None, 
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
        output_name : Optional name of the output TIFF file.
        custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    
    Returns:
        Path: The path of the saved TIFF file.
    """
    save_path = img_reader.img_path.with_name(output_name)

    # Check compression method
    if compression is None:
        current_compression = img_reader.compression_method
    else:
        current_compression = compression

    meta = build_payload(img_reader.metadata,
                         custom_metadata = custom_metadata,
                         compression = compression,)
    meta_write = assemble_tiff_metadata(meta, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    save_tiff(array, save_path, meta_write, compression=current_compression)
    return save_path


def set_channel_labels(img_reader: ImageReader, 
                       channel_labels: str | Sequence[str], 
                       compression: str | None = 'zlib') -> None:
    """
    Set the channel labels in the metadata.
    
    Policy:
    - This function will change the channel labels in metadata, and re-save it with updated metadata.
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
    meta = build_payload(img_reader.metadata,
                         channel_labels=selection.export_labels,
                         compression = compression,)
    meta_write = assemble_tiff_metadata(meta, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    # Get array to save (no z-projection applied here)
    array = img_reader.get_array()
    save_tiff(array, img_reader.img_path, meta_write, compression=compression)  


def apply_zproj(img_reader: ImageReader, 
                z_projection: Zproj = None, 
                compression: str | None = 'zlib') -> None:
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
    
    # Get existing metadata to preserve other fields
    meta = build_payload(img_reader.metadata,
                         z_projection=z_projection,
                         compression = compression,)
    meta_write = assemble_tiff_metadata(meta, 
                                        img_reader.interval, 
                                        img_reader.resolution)
    
    # Get array to save
    array = img_reader.get_array(z_projection)
    save_tiff(array, img_reader.img_path, meta_write, compression=compression)


    
    

    