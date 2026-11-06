from pathlib import Path

import numpy as np
import pytest
from tifffile import TiffFile, imwrite

from fits_io import FitsIO
from fits_io.metadata.channel_colors import conversion_channel_labels
from fits_io.metadata.tiff_meta import ChannelMeta
from fits_io.readers.r_tiff import TiffReader


@pytest.mark.parametrize("labels, colors, expected", [
    ([], ["#00ff00", "#fe0000"], ["green", "red"]),
    (["RFP", ""], ["#00ff00", "#0000ff"], ["RFP", "blue"]),
    ([], ["#00ff00", "#00ff00"], ["green_1", "green_2"]),
    ([], [None, "#ffffff"], ["C1", "white"]),
    (["GFP"], ["#00ff00", "#ff0000"], ["C1", "C2"]),
    ([], ["#00ff00"], ["C1", "C2"]),
    ([], None, ["C1", "C2"]),
    (["green", ""], ["#0000ff", "#00ff00"], ["green", "green_1"]),
])
def test_conversion_defaults(labels: list[str], colors: list[str | None] | None, expected: list[str]) -> None:
    """
    Preserve overrides and resolve blank labels without ambiguous duplicate defaults.
    """
    assert conversion_channel_labels(labels, 2, colors) == expected


def test_numbered_color_labels_keep_label_based_luts() -> None:
    """
    Numbered defaults and white labels retain their display colors in written TIFFs.
    """
    metadata = ChannelMeta(3, ["green_1", "green_2", "white"])
    assert metadata.luts is not None
    assert [lut[:, -1].tolist() for lut in metadata.luts] == [[0, 255, 0], [0, 255, 0], [255, 255, 255]]


def test_user_label_overrides_source_color_in_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep source green metadata independent of the RFP label's red output LUT.
    """
    path = tmp_path / "input.tif"
    imwrite(path, np.ones((8, 8), dtype=np.uint16), imagej=True, metadata={"axes": "YX"})
    monkeypatch.setattr(TiffReader, "channel_colors", property(lambda self: ["#00ff00"]))
    image = FitsIO.from_path(path)
    assert image.channel_colors == ["#00ff00"]
    selection = image.resolve_channel_selection(channel_labels=["RFP"])
    output = image.prepare_conversion(selection=selection)
    target = tmp_path / "converted.tif"
    image.save_array(output.array, metadata=output.metadata, output_path=target)
    with TiffFile(target) as tif:
        assert tif.imagej_metadata is not None
        np.testing.assert_array_equal(np.asarray(tif.imagej_metadata["LUTs"])[:, -1], [255, 0, 0])
