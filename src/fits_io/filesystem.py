from pathlib import Path
import re
from typing import Sequence

from fits_io.image_reader import ImageReader

DEFAULT_SAVE_NAME = "array.tif"


def ends_with_s_number(string: str) -> bool:
    return bool(re.search(r'_s[0-9][0-9]{0,2}$', string))

def get_save_dirs(img_reader: ImageReader) -> Path | list[Path]:
    """Get the output directory paths name to save experiment converted arrays.

    Policy:
    - Always create a per-series folder '<input_stem>_sN', even for a single series (N=1).
    - Check the parent directory of the input file and if it is already containing a '_sN', then return that directory only.

    Args:
        img_reader:
            ImageReader instance describing the input image and its series.

    Returns:
        Path or list of directory paths, one per image series, ready to be written.
    """
    
    nb_arrays = img_reader.series_number
    
    if nb_arrays < 1:
        raise ValueError("ImageReader contains no readable arrays")
    
    # Generate save directories
    base_name = img_reader.img_path.stem
    parent_dir = img_reader.img_path.parent
    
    # Check if the parent directory already contains a '_sN' pattern
    if ends_with_s_number(parent_dir.name):
        return parent_dir
    
    save_dirs: list[Path] = []
    for idx in range(nb_arrays):
        save_dir = parent_dir / f"{base_name}_s{idx+1}"
        save_dirs.append(save_dir)
        
    return save_dirs

def mkdirs_paths(paths: Path | Sequence[Path]) -> list[Path]:
    """Create directories for given path(s) if they do not exist. Retruns the list of created paths.

    Args:
        paths: A single Path or a list of Paths to create directories for.
    """
    if isinstance(paths, Path):
        paths_lst = [paths]
    else:
        paths_lst = list(paths)
        
    for path in paths_lst:
        path.mkdir(parents=True, exist_ok=True)
    return paths_lst

def build_output_path(series_dir: Path, *, save_name: str) -> Path:
    """Return the output file path inside an existing series directory."""
    
    if not isinstance(save_name, str) or not save_name:
        raise ValueError("save_name must be a non-empty string")
    return series_dir / save_name

if __name__ == "__main__":
    s1 = "/data/experiment1_s0235"
    s2 = "/data/experiment1"
    print(ends_with_s_number(s1))  # True
    print(ends_with_s_number(s2))  # False