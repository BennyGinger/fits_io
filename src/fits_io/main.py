from pathlib import Path
from typing import Mapping, Sequence, Any
import logging

import numpy as np

from fits.provenance import is_processed, get_timestamp
from fits_io.image_reader import get_reader
from fits_io.metadata import build_imagej_metadata, STEP_NAME
from fits_io.writer import generate_save_path, save_tiff


logger = logging.getLogger(__name__)


def convert_to_fits_tif(img_path: Path, *, save_path: Path | None = None, channel_labels: str | Sequence[str] | None = None, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib') -> None:
    """Convert an image file to a TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_path : Path to the input image file.
        save_path : Path to save the output TIFF file. If None, saves in a subdirectory next to the input file, by default None
        channel_labels : Channel labels to include in the metadata. If None, uses labels from the input file, by default None
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
    """
    
    # get reader
    reader = get_reader(img_path)
    
    # check if exp was processed/registered
    if is_processed(reader.custom_metadata, step=STEP_NAME):
        timestamp = get_timestamp(reader.custom_metadata, step=STEP_NAME)
        logger.info(f"Image {img_path} has already been processed at {timestamp}. Skipping conversion.")
        return
    
    # read image and get metadata
    meta = build_imagej_metadata(reader, channel_labels=channel_labels, custom_metadata=reader.custom_metadata, extra_step_metadata=user_defined_metadata)
    
    # Generate save path(s)
    save_path_lst = generate_save_path(reader, save_path)
    
    # Get the image array(s)
    arrays = reader.get_array()
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    
    # write tiff with metadata and reader
    if len(arrays) != len(save_path_lst):
        raise ValueError(f"Got {len(arrays)} arrays but {len(save_path_lst)} save paths")

    for array, path in zip(arrays, save_path_lst):
        save_tiff(array, path, meta, compression=compression)


        

if __name__ == '__main__':
    from time import time

    t1 = time()
    
    img_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2.nd2')
    convert_to_fits_tif(img_path, channel_labels=["GFP", "mCherry"])
    print(f"Done in {time()-t1:.2f} seconds")
    
    new_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2_s1/fits_array.tif')
    reader = get_reader(new_path)
    print('____________________')
    print(reader.resolution)
    print(reader.interval)
    print(reader.axes)
    print(reader.channel_labels)
    print(reader.custom_metadata)
    print(type(reader.custom_metadata))
    print(f'Done in {time()-t1:.2f} seconds')
    
    convert_to_fits_tif(new_path)
    print(f"Done in {time()-t1:.2f} seconds")
