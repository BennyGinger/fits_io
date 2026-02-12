from typing import Sequence
import logging

from numpy.typing import NDArray
import numpy as np

from fits_io.readers.factory import ImageReader
from fits_io.readers._types import Zproj


logger = logging.getLogger(__name__)


def get_array_to_export(img_reader: ImageReader, export_channels: list[str], export_all_flag: bool, z_projection: Zproj = None) -> list[NDArray]:
    """
    Get the array(s) to export from the image reader based on the resolved channel labels and export flag. If exporting all channels, retrieves the full array; otherwise, retrieves only the specified channels. Handles cases where the retrieved arrays may be empty and raises an error if so.
    
    Args:
        img_reader: The image reader to retrieve the arrays from.
        export_channels: The resolved channel labels to export.
        export_all_flag: A boolean indicating if all channels are being exported.
        z_projection: An optional Z projection method to apply when retrieving the arrays.
    """
    
    
    if export_all_flag:
        arrays = img_reader.get_array(z_projection)
    else:
        arrays = img_reader.get_channel(export_channels, z_projection)
    
    arrays_list = [arrays] if isinstance(arrays, np.ndarray) else list(arrays)
    if any(a.size == 0 for a in arrays_list):
        raise ValueError("Export produced empty arrays (likely unsupported channel extraction or reader bug).")
    return arrays_list