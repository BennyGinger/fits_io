from pathlib import Path

from tifffile import imwrite
from numpy.typing import NDArray

from fits_io.image_reader import ImageReader
from fits_io.metadata import TiffMetadata



def generate_save_path(img_reader: ImageReader, save_path: Path | None = None) -> list[Path]:
    nb_arrays = img_reader.series_number
    
    if save_path is not None and nb_arrays == 1:
        return [save_path]
    
    # If path were given but multiple arrays, just implement '..._sx.tif' depending on number of arrays
    if save_path is not None and nb_arrays > 1:
        return [save_path.with_stem(f"{save_path.stem}_s{idx+1}") for idx in range(nb_arrays)]
    
    # If no path were given, generate paths
    base_name = img_reader.img_path.stem
    parent_dir = img_reader.img_path.parent
    save_name = 'fits_array.tif'
    
    save_paths: list[Path] = []
    for idx in range(nb_arrays):
        save_dir = parent_dir / f"{base_name}_s{idx+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_paths.append(save_dir / save_name)
        
    return save_paths


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