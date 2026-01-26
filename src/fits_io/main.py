from pathlib import Path
from typing import Mapping, Sequence, Any
import logging

import numpy as np

from fits.provenance import is_processed, get_timestamp
from fits_io.image_reader import get_reader
from fits_io.metadata import build_imagej_metadata, STEP_NAME
from fits_io.writer import generate_save_path, save_tiff


logger = logging.getLogger(__name__)


def convert_to_fits_tif(img_path: Path, *, save_path: Path | None = None, channel_labels: str | Sequence[str] | None = None, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
    """Convert an image file to a TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
    Args:
        img_path : Path to the input image file.
        save_path : Path to save the output TIFF file. If None, saves in a subdirectory next to the input file, by default None
        channel_labels : Channel labels to include in the metadata. If None, uses labels from the input file, by default None
        user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
        compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'
        overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
    """
    
    # get reader
    reader = get_reader(img_path)
    
    # check if exp was processed/registered
    if is_processed(reader.custom_metadata, step=STEP_NAME) and not overwrite:
        timestamp = get_timestamp(reader.custom_metadata, step=STEP_NAME)
        print(f"Image {img_path} has already been processed at {timestamp}. Skipping conversion.")
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

    def load_channel_tiff(path: str | Path, channel_name: str, labels: list[str]) -> np.ndarray:
        """
        Load only one channel from a (<=5D) TIFF hyperstack saved as pages,
        e.g. axes 'TZCYX' with shape (T, Z, C, Y, X).

        Returns: array with axes removed for C -> e.g. 'TZYX' (shape T,Z,Y,X).
        """
        c = labels.index(channel_name)

        with tiff.TiffFile(path) as tif:
            s = tif.series[0]
            axes = s.axes            # e.g. 'TZCYX'
            shape = s.shape          # e.g. (23, 25, 2, 1024, 1024)

            # Split axes into "page axes" (everything except Y and X) + spatial
            page_axes = [ax for ax in axes if ax not in ("Y", "X", "S")]
            page_shape = [shape[axes.index(ax)] for ax in page_axes]

            if "C" not in page_axes:
                raise ValueError(f"No channel axis 'C' found in axes={axes!r}")

            # Build all (t,z,...) combinations with C fixed to c
            # Create a mapping from axis -> index for each combination.
            idx_grids = []
            for ax, n in zip(page_axes, page_shape):
                if ax == "C":
                    idx_grids.append(np.array([c], dtype=int))
                else:
                    idx_grids.append(np.arange(n, dtype=int))

            # Mesh all non-spatial indices, then ravel to page indices
            mesh = np.meshgrid(*idx_grids, indexing="ij")
            multi_idx = np.stack([m.ravel() for m in mesh], axis=0)  # (ndim, n_pages)

            page_indices = np.ravel_multi_index(multi_idx, dims=page_shape, order="C")

            # Read only those pages and reshape back to the page-axes-without-C + YX
            planes = tif.asarray(key=page_indices.tolist())  # shape: (n_pages, Y, X)

            # Figure output shape: all page axes except C, then Y, X
            out_page_axes = [ax for ax in page_axes if ax != "C"]
            out_page_shape = [shape[axes.index(ax)] for ax in out_page_axes]
            y = shape[axes.index("Y")]
            x = shape[axes.index("X")]

            return planes.reshape(*out_page_shape, y, x)
        
    arr = load_channel_tiff(new_path, 'mCherry', ['GFP', 'mCherry'])
    print(f"Loaded array shape: {arr.shape}")
    tiff.imwrite('/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2_s1/mCherry_only.tif', arr)