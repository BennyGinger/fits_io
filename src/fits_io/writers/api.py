from pathlib import Path
from typing import Mapping, Sequence, Any

from numpy.typing import NDArray

from fits_io.metadata.builder_api import meta_orchestration
from fits_io.readers._types import Zproj
from fits_io.readers.protocol import ImageReader
from fits_io.readers.r_tiff import TiffReader
from fits_io.writers.validation import resolve_channel_labels
from fits_io.writers.filesystem import get_save_dirs, build_output_path, mkdirs_paths
from fits_io.writers.core import save_tiff
from fits_io.writers.utils import get_array_to_export

DEFAULT_OUTPUT_NAME = 'fits.tif'

def convert_to_fits_tif(img_reader: ImageReader, *, output_name: str = DEFAULT_OUTPUT_NAME, channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', project_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = 'zlib') -> list[Path]:
    """
    Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_reader : An ImageReader instance for the input image.
        output_name : Optional name of the output TIFF file.
        channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
        export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'
        project_metadata : Additional project-owned metadata to include in the TIFF file, by default None
        z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
    Returns:
        List of Paths of the saved TIFF files.
    """
    # Get the save directories of the image
    save_dirs = get_save_dirs(img_reader)
    
    # Generate save path(s)
    save_dirs = mkdirs_paths(save_dirs)
    save_path_lst = [build_output_path(save_dir, save_name=output_name) for save_dir in save_dirs]
    
    # Set default channel labels to be initialized if user did not provide any
    used_channels, source_channel_indices, export_all_flag = resolve_channel_labels(channel_labels, img_reader.channel_number[0], export_channels)

    # Get the image array(s)
    arrays = get_array_to_export(img_reader, used_channels, export_all_flag, z_projection)
    
    # Write FITS TIFF with metadata and reader
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    for i, (array, path) in enumerate(zip(arrays, save_path_lst)):
        source_channel_count = img_reader.channel_number[i] if i < len(img_reader.channel_number) else img_reader.channel_number[0]
        meta = meta_orchestration(
            img_reader,
            channel_labels=used_channels,
            z_projection=z_projection,
            series_index=i,
            source_channel_indices=source_channel_indices,
            source_channel_count=source_channel_count,
            project_metadata=project_metadata,
            compression=compression,
        )
        # save TIFF
        save_tiff(array, path, meta, compression=compression)
    return save_path_lst

def save_array(img_reader: ImageReader, array: NDArray[Any], axis_order: str, channel_labels: str | Sequence[str] | None, output_name: str = DEFAULT_OUTPUT_NAME, *, project_metadata: Mapping[str, Any] | None = None, compression: str | None = "zlib", ) -> Path:
    """
    Save a given array as a FITS TIFF file with ImageJ metadata, using the input image's path as reference and transfer of metadata.
    
    Note: this function won't apply any z-projection or channel label resolution. It will simply save the provided array with metadata built from the input image and provided parameters.
    
    Args:
        img_reader : An ImageReader instance for the input image, used to access the original image path and metadata.
        array : The image array to be saved as a TIFF file.
        axis_order : Axis order of the input array, used for building correct metadata. Can raise an error in Tifffile if not provided correctly.
        channel_labels : Labels for the channels in the input array, used for building metadata. This should match the channels in the input array and can be a single string for one channel or a sequence of strings for multiple channels.
        output_name : Optional name of the output TIFF file. If None, uses 'fits.tif' by default.
        project_metadata : Additional project-owned metadata to include in the TIFF file, by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default "zlib".
    
    Returns:
        Path of the saved TIFF file.
    """
    save_path = img_reader.img_path.with_name(output_name)

    # Check compression method
    if compression is None:
        current_compression = img_reader.compression_method
    else:
        current_compression = compression

    zproj = img_reader.zproj_method
    metadata = meta_orchestration(
        img_reader,
        channel_labels=channel_labels,
        z_projection=zproj,
        axis_order=axis_order,
        project_metadata=project_metadata,
        compression=current_compression,
    )
    
    save_tiff(array, save_path, metadata, compression=current_compression)
    return save_path

def set_channel_labels(img_reader: ImageReader, channel_labels: str | Sequence[str]) -> None:
    """
    Set the channel labels in the metadata.
    
    Policy:
    - This function will only change the channel labels in metadata, so it will load the existing array and re-save it with updated metadata.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_channel_labels only supports .tif/.tiff files.")
    
    # Get existing metadata to preserve other fields
    meta = meta_orchestration(
        img_reader,
        channel_labels=channel_labels,
        compression=img_reader.compression_method,
    )
    
    array = img_reader.get_array()
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)  

def apply_zproj(img_reader: ImageReader, z_projection: Zproj = None) -> None:
    """
    Apply z-projection to the image array and update the file with the projected array and updated metadata.
    
    Policy:
    - This function will apply the specified z-projection to the existing array and re-save it with updated metadata.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        z_projection : The z-projection method to apply ('max', 'mean', or None). If None, no projection is applied.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("apply_zproj only supports .tif/.tiff files.")
    
    # Get existing metadata to preserve other fields
    meta = meta_orchestration(
        img_reader,
        z_projection=z_projection,
        compression=img_reader.compression_method,
    )
    
    array = img_reader.get_array(z_projection)
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)


if __name__ == '__main__':
    new_path = Path('/media/ben/Analysis/Python/Images/tiff/Run1/c1z25t25v1_tif.tif')
    from fits_io.readers.factory import get_reader
    reader = get_reader(new_path, channel_labels=['GFP'])
    convert_to_fits_tif(reader, channel_labels=['GFP'], export_channels=['GFP'], z_projection='max', compression='zlib')
    
    

    