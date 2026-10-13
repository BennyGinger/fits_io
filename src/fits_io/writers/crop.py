"""Crop spatial axes while retaining all other array dimensions."""

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray


def crop_array_xy(array: NDArray[Any], 
                  axes: str,
                  bounds: tuple[int, int, int, int]
                  ) -> NDArray[Any]:
    """
    Return an XY view for exclusive bounds (x0, y0, x1, y1).
    """
    if len(axes) != array.ndim or axes.count("X") != 1 or axes.count("Y") != 1:
        raise ValueError("XY cropping requires array axes containing exactly one X and Y.")
    
    if len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in bounds):
        raise ValueError("Crop bounds must be four integer pixel coordinates.")
    
    x0, y0, x1, y1 = bounds
    height, width = array.shape[axes.index("Y")], array.shape[axes.index("X")]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Crop bounds {bounds} fall outside the {width} × {height} image.")
    
    selection = [slice(None)] * array.ndim
    selection[axes.index("X")] = slice(x0, x1)
    selection[axes.index("Y")] = slice(y0, y1)
    return array[tuple(selection)]
