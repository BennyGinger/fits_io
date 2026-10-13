import numpy as np
import pytest
import tifffile

from fits_io import FitsIO
from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.tiff_meta import assemble_tiff_metadata
from fits_io.writers.core import save_tiff
from fits_io.writers.crop import crop_array_xy


@pytest.mark.parametrize("axes,shape", [
    ("YX", (9, 10)), ("TZCYX", (2, 3, 2, 9, 10)),
    ("TCZYX", (2, 2, 3, 9, 10)), ("XYC", (10, 9, 2)),
])
def test_crop_uses_named_spatial_axes_and_preserves_other_dimensions(axes, shape):
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    cropped = crop_array_xy(source, axes, (2, 1, 7, 6))
    expected = np.take(np.take(source, range(2, 7), axis=axes.index("X")),
                       range(1, 6), axis=axes.index("Y"))
    np.testing.assert_array_equal(cropped, expected)
    assert np.shares_memory(source, cropped)


@pytest.mark.parametrize("bounds", [(-1, 0, 4, 4), (0, 0, 11, 4), (2, 2, 2, 4),
                                    (0, 0, 4, 10), (0.5, 0, 4, 4), (False, 0, 4, 4)])
def test_invalid_crop_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="Crop bounds"):
        crop_array_xy(np.zeros((9, 10)), "YX", bounds)


def test_crop_save_preserves_artifact_metadata_calibration_and_crop_history(tmp_path):
    raw = np.arange(2 * 3 * 2 * 9 * 10, dtype=np.float32).reshape(2, 3, 2, 9, 10)
    source = tmp_path / "source.tif"
    payload = FitsIOMeta().with_fitsio(
        axes="TZCYX", channel_labels=["GFP", "RFP"],
        source_channel_indices=[2, 4], artifact_channel_indices=[2, 4],
        artifact_type="array", created_by="test").with_custom_metadata({"experiment": "keep"})
    metadata = assemble_tiff_metadata(payload, interval=2.5, resolution=(0.5, 0.25))
    metadata.imagej_meta.update(spacing=3.25, xorigin=10.0, yorigin=20.0, zorigin=2.0)
    save_tiff(raw, source, metadata)
    original_bytes = source.read_bytes()
    reader = FitsIO.from_path(source)
    output = reader.save_array(raw, output_path=tmp_path / "cropped.tif", crop_xy=(2, 1, 7, 6))
    assert source.read_bytes() == original_bytes
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.asarray(), raw[..., 1:6, 2:7])
        assert tif.series[0].axes == "TZCYX"
        assert tif.imagej_metadata["spacing"] == 3.25
        assert tif.imagej_metadata["xorigin"] == 8
        assert tif.imagej_metadata["yorigin"] == 19
        assert tif.imagej_metadata["zorigin"] == 2
        assert tif.imagej_metadata["finterval"] == 2.5
        assert tif.pages[0].tags["XResolution"].value == (2, 1)
        assert tif.pages[0].tags["YResolution"].value == (4, 1)
    cropped_reader = FitsIO.from_path(output)
    meta = cropped_reader.metadata
    assert meta.fits_io.channel_labels == ["GFP", "RFP"]
    assert meta.fits_io.artifact_channel_indices == [2, 4]
    assert meta.fits_io.source_channel_indices == [2, 4]
    assert meta.custom_metadata["experiment"] == "keep"
    assert meta.custom_metadata["spatial_crops"] == [{
        "source_axes": "TZCYX", "source_shape": list(raw.shape),
        "bounds_xy": [2, 1, 7, 6], "offset_xy": [2, 1]}]
    again = cropped_reader.save_array(cropped_reader.get_array().array,
                                      output_path=tmp_path / "again.tif", crop_xy=(1, 1, 4, 4))
    assert len(FitsIO.from_path(again).metadata.custom_metadata["spatial_crops"]) == 2
    with tifffile.TiffFile(again) as tif:
        assert tif.imagej_metadata["xorigin"] == 7
        assert tif.imagej_metadata["yorigin"] == 18


def test_explicit_metadata_is_preserved_when_cropping(tmp_path):
    source = tmp_path / "source.tif"
    raw = np.ones((8, 9), dtype=np.uint16)
    tifffile.imwrite(source, raw)
    reader = FitsIO.from_path(source)
    payload = reader.build_payload(export_channels="all", custom_metadata={"display": "keep"})
    output = reader.save_array(raw, metadata=payload, crop_xy=(1, 2, 6, 7),
                               output_path=tmp_path / "export.tif")
    metadata = FitsIO.from_path(output).metadata
    assert metadata.custom_metadata["display"] == "keep"
    assert metadata.custom_metadata["spatial_crops"][0]["bounds_xy"] == [1, 2, 6, 7]
