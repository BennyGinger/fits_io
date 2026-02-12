from pathlib import Path
from typing import Type
import logging

from fits_io.readers.r_nd2 import Nd2Reader
from fits_io.readers.r_tiff import TiffReader
import nd2
import numpy as np

from fits_io.readers.protocol import ImageReader


logger = logging.getLogger(__name__)

class ImageReaderError(Exception):
    """Base exception for reader-related errors."""

class ReaderFileNotFoundError(ImageReaderError):
    pass

class UnsupportedFileTypeError(ImageReaderError):
    pass


READER_BY_SUFFIX: dict[str, Type[ImageReader]] = {
    ".tif": TiffReader,
    ".tiff": TiffReader,
    ".nd2": Nd2Reader,
}

def get_reader(path: str | Path, channel_labels: list[str] | None = None) -> ImageReader:
    """Return an ImageReader instance for the given path, based on suffix."""
    p = Path(path)
    
    if not p.exists() or not p.is_file():
        raise ReaderFileNotFoundError(f"Path not found or is not a file: {p}")
    
    suffix = p.suffix.lower()

    try:
        reader_cls = READER_BY_SUFFIX[suffix]
    except KeyError as e:
        supported = ", ".join(sorted(READER_BY_SUFFIX))
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix!r}. Supported: {supported}"
        ) from e

    return reader_cls(p, _channel_labels=channel_labels)



if __name__ == "__main__":
    
    # tif_path1 = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    # stack1 = imread(tif_path1)
    # print(stack1.shape)
    # tif_path2 = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run4/c4z1t91v1.tif')
    # stack2 = imread(tif_path2)
    # print(stack2.shape)
    
    # with TiffWriter('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif') as tif:
    #     tif.write(stack1, metadata={'axes':'TZCYX'}, compression=None)
    #     tif.write(stack2, metadata={'axes':'TCYX'}, compression='lzw', photometric="minisblack", planarconfig="separate")
    
    # with TiffFile("/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif") as tif:
    #     print("n_series:", len(tif.series))
    #     for i, s in enumerate(tif.series):
    #         print(i, "axes:", s.axes, "shape:", s.shape, "n_pages:", len(s.pages))
    
    with nd2.ND2File("/media/ben/Analysis/Python/Docker_mount/Test_images/nd2/Run2_test/control/c2z25t23v1_nd2.nd2") as f:
        darr = f.to_dask()
        print(type(darr))
        print(darr.shape)
        
        print(f.sizes)
        axes = tuple(f.sizes.keys())
        print(axes)
        c_axis = axes.index('C')
        print("c_axis:", c_axis)
        
        ch_idx = [0]
        
        slicer = [slice(None)] * darr.ndim
        slicer[c_axis] = ch_idx
        
        chan_arr = np.squeeze(darr[tuple(slicer)].compute())
        print(type(chan_arr), chan_arr.shape)
        

        # example: take channel 0
        # (axis order depends on the file)
        
    