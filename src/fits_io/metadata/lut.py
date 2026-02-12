import numpy as np
from numpy.typing import NDArray

COLOR_MAP = {
        "red":     (1, 0, 0),
        "green":   (0, 1, 0),
        "blue":    (0, 0, 1),
        "cyan":    (0, 1, 1),
        "magenta": (1, 0, 1),
        "yellow":  (1, 1, 0),
        "gray":    (1, 1, 1),
    }

LABEL_TO_COLOR = {
    "blue": 'blue',
    "bfp": 'blue',
    "dapi": 'blue',
    "cyan": 'cyan',
    "cfp": 'cyan',
    "yellow": 'yellow',
    "cy3": 'yellow',
    "green": 'green',
    "gcamp": 'green',
    "gfp": 'green',
    "egfp": 'green',
    "fitc": 'green',
    "magenta": 'magenta',
    "mch": 'magenta',
    "ired": 'magenta',
    "irfp": 'magenta',
    "red": 'red',
    "pinky": 'red',
    "mkate2": 'red',
    "scarlet": 'red',
    "geco": 'red',
    "mcherry": 'red',
    "tritc": 'red',
    "rfp": 'red',
    'gray': 'gray',
    'grey': 'gray',
}

def make_color_lut(color: str) -> NDArray[np.uint8]:
    """Return ImageJ-style LUT: shape (3, 256), uint8."""
    
    try:
        mask = np.array(COLOR_MAP[color.lower()], dtype=np.uint8)[:, None]
    except KeyError:
        raise ValueError(f"Unsupported LUT color: {color}")

    return mask * np.arange(256, dtype=np.uint8)   