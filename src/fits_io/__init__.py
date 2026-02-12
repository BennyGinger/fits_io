from fits_io.client import FitsIO
from fits_io.readers._types import ExtTags

SUPPORTED_EXTENSIONS: set[ExtTags] = {'.tiff', '.tif', '.nd2'}

__all__ = [
    "FitsIO",
    "SUPPORTED_EXTENSIONS",
]