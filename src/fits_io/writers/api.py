from pathlib import Path
from typing import Mapping, Sequence, Any
import logging

from numpy.typing import NDArray

from fits_io.metadata.context import resolve_build_context
from fits_io.metadata.imagej import build_tiff_metadata
from fits_io.metadata.private import build_private_payload
from fits_io.readers._types import StatusFlag, Zproj
from fits_io.readers.protocol import ImageReader, ALLOWED_FLAGS
from fits_io.readers.factory import get_reader
from fits_io.readers.r_tiff import TiffReader
from fits_io.writers.validation import resolve_channel_labels
from fits_io.writers.filesystem import get_save_dirs, build_output_path, mkdirs_paths
from fits_io.writers.core import save_tiff
from fits_io.writers.utils import get_array_to_export


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = 'fits.tif'




def convert_to_fits_tif(img_reader: ImageReader, *, user_name: str = 'unknown', distribution: str | None = None, step_name: str | None = None, output_name: str = DEFAULT_OUTPUT_NAME, channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', custom_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = 'zlib') -> list[Path]:
    """
    Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_reader : An ImageReader instance for the input image.
        user_name : Name of the user performing the conversion, by default 'unknown'
        distribution : Name of the distribution or package, by default None
        step_name : Name of the processing step, by default None
        output_name : Optional name of the output TIFF file.
        channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
        export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'
        custom_metadata : Additional custom metadata to include in the TIFF file, by default None
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
    used_channels, export_all_flag = resolve_channel_labels(channel_labels, img_reader.channel_number[0], export_channels)
    
    # Get the image array(s)
    arrays = get_array_to_export(img_reader, used_channels, export_all_flag, z_projection)
    
    # Write FITS TIFF with metadata and reader
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    for i, (array, path) in enumerate(zip(arrays, save_path_lst)):
        ctx = resolve_build_context(img_reader, step_name=step_name, channel_labels=used_channels, z_projection=z_projection, new_user=user_name, series_index=i)
        payload = build_private_payload(ctx, distribution=distribution, extra_step_metadata=custom_metadata, z_projection=z_projection)
        meta = build_tiff_metadata(ctx, payload)
        # save TIFF
        save_tiff(array, path, meta, compression=compression)
    return save_path_lst

def save_array(img_reader: ImageReader, array: NDArray[Any], axis_order: str, channel_labels: str | Sequence[str], output_name: str = DEFAULT_OUTPUT_NAME, *, user_name: str = "unknown", distribution: str | None = None, step_name: str | None = None, custom_metadata: Mapping[str, Any] | None = None, compression: str | None = "zlib", ) -> Path:
    """
    Save a given array as a FITS TIFF file with ImageJ metadata, using the input image's path as reference and transfer of metadata.
    
    Note: this function won't apply any z-projection, channel label resolution, or provenance tag management. It will simply save the provided array with metadata built from the input image and provided parameters.
    
    Args:
        img_reader : An ImageReader instance for the input image, used to access the original image path and metadata.
        array : The image array to be saved as a TIFF file.
        axis_order : Axis order of the input array, used for building correct metadata. Can raise an error in Tifffile if not provided correctly.
        channel_labels : Labels for the channels in the input array, used for building metadata. This should match the channels in the input array and can be a single string for one channel or a sequence of strings for multiple channels.
        output_name : Optional name of the output TIFF file. If None, uses 'fits.tif' by default.
        user_name : Name of the user performing the save operation, by default "unknown".
        distribution : Name of the distribution or package, by default None.
        step_name : Name of the processing step, by default None.
        custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default "zlib".
    
    Returns:
        Path of the saved TIFF file.
    """
    save_path = img_reader.img_path.with_name(output_name)

    zproj = img_reader.zproj_method
    ctx = resolve_build_context(img_reader, step_name=step_name, channel_labels=channel_labels, z_projection=zproj, new_user=user_name, axis_order=axis_order)
    payload = build_private_payload(ctx, distribution=distribution, extra_step_metadata=custom_metadata, z_projection=zproj)
    metadata = build_tiff_metadata(ctx, payload)

    # Check comperssion method
    if compression is None:
        current_compression = img_reader.compression_method
    else:
        current_compression = compression
    
    save_tiff(array, save_path, metadata, compression=current_compression)
    return save_path

def set_status(img_reader: ImageReader, status: StatusFlag) -> None:
    """
    Set the status of the image to either 'active' or 'skip'.
    
    Policy:
    - This function will only change the status in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, channel labels, compression or provenance tag is applied here.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        status : The status string to set.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_status only supports .tif/.tiff files.")
    
    if status not in ALLOWED_FLAGS:
        raise ValueError(f"Invalid status: {status}. Must be one of {ALLOWED_FLAGS}")
    
    ctx = resolve_build_context(img_reader, new_status=status)
    payload = build_private_payload(ctx, add_step_meta=False)
    meta = build_tiff_metadata(ctx, payload)
    
    array = img_reader.get_array()
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)

def set_channel_labels(img_reader: ImageReader, channel_labels: str | Sequence[str]) -> None:
    """
    Set the channel labels in the metadata.
    
    Policy:
    - This function will only change the channel labels in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, change status, compression or provenance tag is applied here.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_channel_labels only supports .tif/.tiff files.")
    
    # Get existing metadata to preserve other fields
    ctx = resolve_build_context(img_reader, channel_labels=channel_labels)
    payload = build_private_payload(ctx, add_step_meta=False)
    meta = build_tiff_metadata(ctx, payload)
    
    array = img_reader.get_array()
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)  

def apply_zproj(img_reader: ImageReader, z_projection: Zproj = None) -> None:
    """
    Apply z-projection to the image array and update the file with the projected array and updated metadata.
    
    Policy:
    - This function will apply the specified z-projection to the existing array in the file, and re-save it with updated metadata. So, no change to channel labels, status, compression or provenance tag is applied here.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        z_projection : The z-projection method to apply ('max', 'mean', or None). If None, no projection is applied.
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("apply_zproj only supports .tif/.tiff files.")
    
    # Get existing metadata to preserve other fields
    ctx = resolve_build_context(img_reader, z_projection=z_projection)
    payload = build_private_payload(ctx, add_step_meta=False, z_projection=z_projection)
    meta = build_tiff_metadata(ctx, payload)
    
    array = img_reader.get_array(z_projection)
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)


if __name__ == '__main__':
    from time import time
    import tifffile as tiff
    from fits_io.readers.factory import get_reader

    t1 = time()
    
    
    new_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    reader = get_reader(new_path)
    convert_to_fits_tif(reader)
    
    

    