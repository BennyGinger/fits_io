"""ImageJ TIFF channel colors are available without decoding image pixels."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from tifffile import TiffPage, imwrite

from fits_io import FitsIO
from fits_io.metadata.imageJ_meta import make_color_lut
from fits_io.readers.r_tiff import TiffReader


@pytest.mark.parametrize("channels", [1, 4])
def test_cached_imagej_colors_are_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               channels: int,
                                               ) -> None:
    """
    Read both single-LUT and multiple-LUT encodings, preserving a nonstandard red value.
    """
    path = tmp_path / "raw.tif"
    luts = [make_color_lut(color) for color in ["blue", "green", "red", "gray"][:channels]]
    if channels == 4:
        luts[2][0, -1] = 254
    imwrite(path, np.zeros((channels, 5, 7), dtype=np.uint16), imagej=True,
            metadata={"axes": "CYX", "LUTs": luts})

    def forbid_pixels(*args: object, **kwargs: object) -> None:
        """
        Ensure channel metadata does not require a TIFF page read.
        """
        pytest.fail("Color lookup decoded image pixels")

    monkeypatch.setattr(TiffPage, "asarray", forbid_pixels)
    image = FitsIO.from_path(path)
    assert image.channel_colors == ["#0000ff", "#00ff00", "#fe0000", "#ffffff"][:channels]
    assert image.resolve_channel_selection().source_labels == ["blue", "green", "red", "white"][:channels]


@pytest.mark.parametrize("luts, expected", [
    (None, None),
    ([make_color_lut("green")], None),
    (np.zeros((2, 3, 256), dtype=np.uint8), [None, None]),
    (np.zeros((2, 256, 3), dtype=np.uint8), None),
    (np.full((2, 3, 256), 256, dtype=np.uint16), [None, None]),
    (np.zeros((2, 3, 256), dtype=float), None),
])
def test_unusable_luts_return_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      luts: list[NDArray[np.uint8]] | NDArray | None,
                                      expected: list[str | None] | None,
                                      ) -> None:
    """
    Reject missing, black, malformed and channel-count-mismatched LUT metadata.
    """
    path = tmp_path / "raw.tif"
    imwrite(path, np.zeros((2, 5, 7), dtype=np.uint16), imagej=True, metadata={"axes": "CYX"})
    reader = TiffReader(path)
    monkeypatch.setattr(reader, "_imageJ_meta", {"LUTs": luts})
    assert reader.channel_colors == expected
