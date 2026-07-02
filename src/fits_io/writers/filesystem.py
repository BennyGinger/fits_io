from pathlib import Path
import re
from typing import Sequence

from fits_io.readers.factory import ImageReader

DEFAULT_SAVE_NAME = "array.tif"


def _ends_with_s_number(string: str) -> bool:
    return bool(re.search(r'_s[0-9][0-9]{0,2}$', string))

def get_save_dirs(series_reader: list[ImageReader]) -> list[Path]:
    """Get the output directory paths name to save experiment converted arrays.

    Policy:
    - Always create a per-series folder '<input_stem>_sN', even for a single series (N=1).
    - If file is already a fits file then parent directory already contains a '_sN' pattern. In that case return the parent directory as is.

    Args:
        img_reader:
            ImageReader instance describing the input image and its series.

    Returns:
        list[Path]: List of directory paths, one per image series, ready to be written.
    """
    save_dirs: list[Path] = []
    for img_reader in series_reader:
        if not isinstance(img_reader, ImageReader):
            raise TypeError(f"Expected ImageReader instance, got {type(img_reader)}")
        # Generate save directories
        base_name = img_reader.img_path.stem
        parent_dir = img_reader.img_path.parent
        
        # If files already a fits file then parent directory already contains a '_sN' pattern
        if _ends_with_s_number(parent_dir.name):
            save_dirs.append(parent_dir)
            continue
        
        # Otherwise create per-series directories
        idx = img_reader.series_idx
        save_dir = parent_dir / f"{base_name}_s{idx+1}"
        save_dirs.append(save_dir)      
    return save_dirs

def mkdirs_paths(paths: Path | Sequence[Path]) -> list[Path]:
    """Create directories for given path(s) if they do not exist. Retruns the list of created paths.

    Args:
        paths: A single Path or a list of Paths to create directories for.
    """
    paths_lst = [paths] if isinstance(paths, Path) else list(paths)
    
    for path in paths_lst:
        path.mkdir(parents=True, exist_ok=True)
    return paths_lst

def build_output_paths(series_reader: list[ImageReader], save_name: str) -> list[Path]:
    """
    Build the output file path(s) for saving the converted FITS TIFF files.
    
    Args:
        series_reader: List of ImageReader instances, one per series.
        save_name: Name of the output file to save in each series directory.
    
    Returns:
        List of Paths for the output files, one per series.
    """
    series_dir = get_save_dirs(series_reader)
    
    for dir_path in series_dir:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if not isinstance(save_name, str) or not save_name:
        raise ValueError("save_name must be a non-empty string")
    return [dir_path / save_name for dir_path in series_dir]

if __name__ == "__main__":
    s1 = "/data/experiment1_s0235"
    s2 = "/data/experiment1"
    print(_ends_with_s_number(s1))  # True
    print(_ends_with_s_number(s2))  # False