import numpy as np
import tifffile

from fits_io import FitsIO


def test_save_array_embeds_overlay_bytes_without_changing_pixels(tmp_path):
    source = tmp_path / "source.tif"
    raw = np.arange(2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 3, 8, 9)
    tifffile.imwrite(source, raw, imagej=True, metadata={"axes": "TCYX"})
    reader = FitsIO.from_path(source)
    # Serialization/interpretation belongs to the caller; fits_io carries the bytes.
    overlays = [b"first serialized ROI", b"second serialized ROI"]
    output = reader.save_array(
        raw, output_path=tmp_path / "overlay.tif", imagej_overlays=overlays,
        compression="none",
        custom_metadata={"purpose": "overlay round trip"})
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.asarray(), raw)
        assert tif.series[0].axes == "TCYX"
        assert tif.imagej_metadata["Overlays"] == overlays
    assert FitsIO.from_path(output).metadata.custom_metadata["purpose"] == "overlay round trip"


def test_empty_overlay_remains_a_normal_imagej_tiff(tmp_path):
    source = tmp_path / "source.tif"
    raw = np.zeros((8, 9), dtype=np.uint8)
    tifffile.imwrite(source, raw)
    output = FitsIO.from_path(source).save_array(
        raw, output_path=tmp_path / "without_overlay.tif", imagej_overlays=[])
    with tifffile.TiffFile(output) as tif:
        assert "Overlays" not in tif.imagej_metadata
        np.testing.assert_array_equal(tif.asarray(), raw)
