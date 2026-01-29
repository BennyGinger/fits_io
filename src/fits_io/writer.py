from pathlib import Path
from typing import Mapping, Sequence, Any
import logging

from numpy.typing import NDArray
import numpy as np
from tifffile import imwrite

from fits_io.provenance import is_processed, get_timestamp, ExportProfile
from fits_io.image_reader import ImageReader
from fits_io.metadata import build_imagej_metadata, TiffMetadata
from fits_io.filesystem import get_save_dirs, build_output_path, mkdirs_paths

STEP_NAME = 'fits_io.convert'
DIST_NAME = "fits-io"
FILENAME = "fits_array.tif"
EXPORT_PROFILE = ExportProfile(dist_name=DIST_NAME, step_name=STEP_NAME, filename=FILENAME)

logger = logging.getLogger(__name__)


def _save_tiff(img_array: NDArray, save_path: Path, metadata: TiffMetadata, compression: str | None = 'zlib') -> None:
    
    predictor = 2 if compression in {"zlib", "deflate", "lzma"} else None
    
    imwrite(save_path,
            img_array,
            imagej=True,
            metadata=metadata.imagej_meta,
            resolution=metadata.resolution,
            predictor=predictor,
            extratags=metadata.extratags,
            compression=compression,
    )

def convert_to_fits_tif(img_reader: ImageReader, *, channel_labels: str | Sequence[str] | None = None, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
    """Convert an image file to a TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_reader : An ImageReader instance for the input image.
        channel_labels : Channel labels to include in the metadata. If None, uses labels from the input file, by default None
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
        overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
    """
    
    # check if exp was processed/registered
    if is_processed(img_reader.custom_metadata, step=STEP_NAME) and not overwrite:
        timestamp = get_timestamp(img_reader.custom_metadata, step=STEP_NAME)
        print(f"Image {img_reader.img_path} has already been processed at {timestamp}. Skipping conversion.")
        logger.info(f"Image {img_reader.img_path} has already been processed at {timestamp}. Skipping conversion.")
        return
    
    # set default channel labels to be initialized if user did not provide any
    if channel_labels is None:
        channel_labels = 'initialize'
    
    # get metadata
    meta = build_imagej_metadata(img_reader,
                                 export_profile=EXPORT_PROFILE,
                                 channel_labels=channel_labels, 
                                 extra_step_metadata=user_defined_metadata)
    
    # Generate save path(s)
    save_dirs = get_save_dirs(img_reader)
    save_dirs = mkdirs_paths(save_dirs)
    save_path_lst = [build_output_path(save_dir, save_name=FILENAME) for save_dir in save_dirs]
    
    # Get the image array(s)
    arrays = img_reader.get_array()
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    
    # write tiff with metadata and reader
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    for array, path in zip(arrays, save_path_lst):
        _save_tiff(array, path, meta, compression=compression)

def save_fits_array(img_reader: ImageReader, *, export_profile: ExportProfile, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
    """Save the FITS array from an ImageReader instance to a TIFF file with ImageJ metadata.
    
    Args:
        img_reader : An ImageReader instance for the input image.
        dist_name : Name of the distribution or package.
        step_name : Name of the processing step.
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
        overwrite : If True, overwrite existing files. If False and the output file exists, skip saving, by default False
    """
    # check if exp was processed/registered
    if is_processed(img_reader.custom_metadata, step=export_profile.step_name) and not overwrite:
        timestamp = get_timestamp(img_reader.custom_metadata, step=export_profile.step_name)
        print(f"Image {img_reader.img_path} has already been processed at {timestamp}. Skipping conversion.")
        logger.info(f"Image {img_reader.img_path} has already been processed at {timestamp}. Skipping conversion.")
        return

    # get metadata
    meta = build_imagej_metadata(img_reader, 
                                 export_profile=export_profile,
                                 extra_step_metadata=user_defined_metadata)
    
    # Generate save path
    save_dir = img_reader.img_path.parent
    save_path = build_output_path(save_dir, save_name=export_profile.filename)
    array = img_reader.get_array()
    
    if isinstance(array, list):
        raise ValueError("Expected a single array, but got multiple series. You may need to use convert_to_fits_tif instead.")
    
    _save_tiff(array, save_path, meta, compression=compression)


        

if __name__ == '__main__':
    from time import time
    import tifffile as tiff

    t1 = time()
    
    img_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2.nd2')
    # convert_to_fits_tif(img_path, channel_labels=["GFP", "mCherry"])
    # print(f"Done in {time()-t1:.2f} seconds")
    
    new_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2_s1/fits_array.tif')
    # reader = get_reader(new_path)
    # print('____________________')
    # print(reader.resolution)
    # print(reader.interval)
    # print(reader.axes)
    # print(reader.channel_labels)
    # print(reader.custom_metadata)
    # print(type(reader.custom_metadata))
    # print(f'Done in {time()-t1:.2f} seconds')
    
    # convert_to_fits_tif(new_path)
    # print(f"Done in {time()-t1:.2f} seconds")

    