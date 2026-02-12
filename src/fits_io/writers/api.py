from pathlib import Path
from typing import Mapping, Sequence, Any
import logging
from functools import partial

from fits_io.readers._types import StatusFlag, Zproj
from fits_io.readers.protocol import ImageReader, ALLOWED_FLAGS
from fits_io.readers.factory import get_reader
from fits_io.metadata.builder import build_metadata
from fits_io.readers.r_tiff import TiffReader
from fits_io.writers.validation import resolve_channel_labels, image_converted
from fits_io.writers.filesystem import get_save_dirs, build_output_path, mkdirs_paths
from fits_io.writers.core import save_tiff
from fits_io.writers.utils import get_array_to_export


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = 'fits.tif'



def convert_to_fits_tif(img_reader: ImageReader, *, user_name: str = 'unknown', distribution: str | None = None, step_name: str | None = None, output_name: str = DEFAULT_OUTPUT_NAME, expected_filenames: set[str] | None = None, channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', user_defined_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = 'zlib', overwrite: bool = False) -> list[Path]:
    """
    Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_reader : An ImageReader instance for the input image.
        user_name : Name of the user performing the conversion, by default 'unknown'
        distribution : Name of the distribution or package, by default None
        step_name : Name of the processing step, by default None
        output_name : Optional name of the output TIFF file.
        expected_filenames : Optional set of expected output filenames to check for existing converted files, e.g. {"array.tif", "array_zproj.tif"}. If None, defaults to {output_name}.
        channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
        export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
        overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
    Returns:
        List of Paths of the saved TIFF files.
    """
    # Get the save directories of the image
    save_dirs = get_save_dirs(img_reader)
    
    # check if exp was processed/registered
    expected_filenames = expected_filenames or {output_name}
    if image_converted(save_dirs, expected_filenames) and not overwrite:
        logger.info(f"Image {img_reader.img_path} has already been converted. Skipping conversion.")
        return save_dirs
    
    # Generate save path(s)
    save_dirs = mkdirs_paths(save_dirs)
    save_path_lst = [build_output_path(save_dir, save_name=output_name) for save_dir in save_dirs]
    
    # Set default channel labels to be initialized if user did not provide any
    used_channels, export_all_flag = resolve_channel_labels(channel_labels, img_reader.channel_number[0], export_channels)
    
    # Get the image array(s)
    arrays = get_array_to_export(img_reader, used_channels, export_all_flag, z_projection)
    
    # Prepare metadata
    build_md = partial(build_metadata,
                        img_reader,
                        user_name=user_name,
                        distribution=distribution,
                        step_name=step_name,
                        channel_labels=used_channels,
                        z_projection=z_projection,
                        extra_step_metadata=user_defined_metadata)
    
    # Write FITS TIFF with metadata and reader
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    for i, (array, path) in enumerate(zip(arrays, save_path_lst)):
        # finish building metadata
        meta = build_md(series_index=i)
        # save TIFF
        save_tiff(array, path, meta, compression=compression)
    return save_path_lst

def save_fits_array(img_reader: ImageReader, *, distribution: str | None = None, step_name: str | None = None, output_name: str = DEFAULT_OUTPUT_NAME, user_name: str = 'unknown', user_defined_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = 'zlib') -> None:
    """Save the FITS array from an ImageReader instance to a TIFF file with ImageJ metadata.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        distribution : Name of the distribution or package.
        step_name : Name of the processing step.
        output_name : Optional name of the output TIFF file. If None, uses 'fits.tif' by default.
        user_name : Name of the user performing the save, by default 'unknown'
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
    """
    # get metadata
    meta = build_metadata(img_reader, 
                                 distribution=distribution,
                                 step_name=step_name,
                                 user_name=user_name,
                                 extra_step_metadata=user_defined_metadata)
    
    # Generate save path
    save_dir = img_reader.img_path.parent
    save_path = build_output_path(save_dir, save_name=output_name)
    array = img_reader.get_array(z_projection)
    
    if isinstance(array, list):
        raise ValueError("Multiple series detected; use convert_to_fits_tif.")
    
    save_tiff(array, save_path, meta, compression=compression)

def set_status(img_reader: ImageReader, status: StatusFlag, user_name: str = 'unknown') -> None:
    """
    Set the status of the image to either 'active' or 'skip'.
    
    Policy:
    - This function will only change the status in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, channel labels, compression or provenance tag is applied here.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        status : The status string to set.
        user_name : Name of the user setting the status, by default 'unknown'
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_status only supports .tif/.tiff files.")
    
    if status not in ALLOWED_FLAGS:
        raise ValueError(f"Invalid status: {status}. Must be one of {ALLOWED_FLAGS}")
    
    meta = build_metadata(img_reader,
                                 user_name=user_name,
                                 add_step_meta=False, # do not add step metadata for status change
                                 new_status=status)
    
    array = img_reader.get_array()
    
    compression = img_reader.compression_method
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    save_tiff(array, img_reader.img_path, meta, compression=compression)

def set_channel_labels(img_reader: ImageReader, channel_labels: str | Sequence[str], user_name: str = 'unknown') -> None:
    """
    Set the channel labels in the metadata.
    
    Policy:
    - This function will only change the channel labels in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, change status, compression or provenance tag is applied here.
    - Multi-series inputs are not supported here by design.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
        user_name : Name of the user setting the channel labels, by default 'unknown'
    """
    if not isinstance(img_reader, TiffReader):
        raise TypeError("set_channel_labels only supports .tif/.tiff files.")
    
    # Get existing metadata to preserve other fields
    meta = build_metadata(img_reader,
                                 user_name=user_name,
                                 channel_labels=channel_labels,
                                 add_step_meta=False) # do not add step metadata for metadata update
    
    array = img_reader.get_array()
    
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
    
    

    