from pathlib import Path

from tifffile import imwrite
from numpy.typing import NDArray

from fits_io.metadata import TiffMetadata



def save_tiff(img_array: NDArray, save_path: Path, metadata: TiffMetadata, compression: str | None = 'zlib') -> None:
    
    predictor = 2
    if compression is None:
        predictor = None
    
    imwrite(save_path,
            img_array,
            imagej=True,
            metadata=metadata.imagej_meta,
            resolution=metadata.resolution,
            predictor=predictor,
            extratags=metadata.extratags,
            compression=compression,
    )