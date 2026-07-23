from pathlib import Path
import re
from typing import Sequence

from fits_io.readers.factory import ImageReader

DEFAULT_SAVE_NAME = "array.tif"


def _ends_with_s_number(string: str) -> bool:
    return bool(re.search(r'_s[0-9][0-9]{0,2}$', string))


def get_save_dir(img_reader: ImageReader) -> Path:
    """
    Return the directory where this series should be written.
    """
    base_name = img_reader.img_path.stem
    parent_dir = img_reader.img_path.parent

    # Already inside a FITS series directory
    if _ends_with_s_number(parent_dir.name):
        return parent_dir

    return parent_dir / f"{base_name}_s{img_reader.series_idx + 1}"


def build_output_path(
    img_reader: ImageReader,
    save_name: str,
) -> Path:
    """
    Build the output file path for one image series.
    """
    if not save_name:
        raise ValueError("save_name must be a non-empty string")

    save_dir = get_save_dir(img_reader)
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / save_name


def create_save_path(base_path: Path, output_name: str,) -> Path:
    parent = base_path if base_path.is_dir() else base_path.parent
    return parent / output_name


if __name__ == "__main__":
    s1 = "/data/experiment1_s0235"
    s2 = "/data/experiment1"
    print(_ends_with_s_number(s1))  # True
    print(_ends_with_s_number(s2))  # False