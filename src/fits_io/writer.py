from pathlib import Path

from tifffile import imwrite
from numpy.typing import NDArray

from fits_io.image_reader import ImageReader
from fits_io.metadata import TiffMetadata

DEFAULT_SAVE_NAME = "array.tif"

def generate_convert_save_paths(img_reader: ImageReader, *, save_name: str) -> list[Path]:
    """Generate output file paths for image conversion.

    Policy:
    - Always create a per-series folder '<input_stem>_sN', even for a single series (N=1).
    - Output folders are created next to the input image path.
    - The same `save_name` is used inside each per-series folder.

    This function is intended for use during the initial conversion step.
    Downstream pipeline stages should rely on existing series folders and
    use `build_output_path` to construct output file paths.

    Args:
        img_reader:
            ImageReader instance describing the input image and its series.
        save_name:
            Output filename to use inside each per-series folder
            (e.g. 'fits_array.tif', 'fits_masks.tif').

    Returns:
        List of file paths, one per image series, ready to be written.
    """

    if not save_name:
        raise ValueError("save_name must be a non-empty string")
    
    nb_arrays = img_reader.series_number
    
    if nb_arrays < 1:
        raise ValueError("ImageReader contains no readable arrays")
    
    # Generate save paths
    base_name = img_reader.img_path.stem
    parent_dir = img_reader.img_path.parent
    
    save_paths: list[Path] = []
    for idx in range(nb_arrays):
        save_dir = parent_dir / f"{base_name}_s{idx+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = build_output_path(save_dir, save_name=save_name)
        save_paths.append(out_path)
        
    return save_paths

def build_output_path(series_dir: Path, *, save_name: str) -> Path:
    """Return the output file path inside an existing series directory."""
    return series_dir / save_name

def save_tiff(img_array: NDArray, save_path: Path, metadata: TiffMetadata, compression: str | None = 'zlib') -> None:
    
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