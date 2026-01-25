from pathlib import Path
from typing import Any

from tifffile import imwrite

from fits_io.metadata import TiffMetadata



def save_tiff(img_array: Any, save_path: Path, metadata: TiffMetadata, compression: str | None = 'zlib') -> None:
    imwrite(save_path,
            img_array,
            imagej=True,
            metadata=metadata.imagej_meta,
            resolution=metadata.resolution,
            predictor=2,
            extratags=metadata.extratags,
            compression=compression,
    )