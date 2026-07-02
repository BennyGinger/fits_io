import numpy as np
import pytest

from fits_io.metadata.imageJ_meta import make_color_lut

# -------------------------
# make_color_lut
# -------------------------

@pytest.mark.parametrize("color, idx", [("red", 0), ("green", 1), ("blue", 2)])
def test_make_color_lut_shape_dtype_and_ramp(color: str, idx: int):
    lut = make_color_lut(color)
    assert lut.shape == (3, 256)
    assert lut.dtype == np.uint8
    assert np.array_equal(lut[idx], np.arange(256, dtype=np.uint8))
    for j in {0, 1, 2} - {idx}:
        assert np.all(lut[j] == 0)