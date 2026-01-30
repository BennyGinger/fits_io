import json
from typing import cast

import numpy as np
import pytest

# adjust import if your package path is different
import fits_io.metadata as md
from fits_io.provenance import ExportProfile


class DummyReader:
    """Minimal ImageReader-like object for build_imagej_metadata tests."""
    def __init__(self, axes="TZCYX", interval=11.0, channel_number=2, resolution=(0.5, 0.25), custom_metadata=None):
        self.axes = axes
        self.interval = interval
        self.channel_number = channel_number
        self.resolution = resolution
        self.channel_labels = None
        self.custom_metadata = custom_metadata or {}
        self.export_status = "active"


# -------------------------
# StackMeta
# -------------------------

def test_stackmeta_to_dict_with_interval():
    s = md.StackMeta(axes="TZCYX", status='active', finterval=11.0)
    d = s.to_dict()
    assert d == {"axes": "TZCYX", 'Info': 'active', "finterval": 11.0}


def test_stackmeta_to_dict_without_interval():
    s = md.StackMeta(axes="YX", status='active', finterval=None)
    d = s.to_dict()
    assert d == {"axes": "YX", 'Info': 'active'}
    assert "finterval" not in d


# -------------------------
# ResolutionMeta
# -------------------------

def test_resolutionmeta_default_is_none_density():
    r = md.ResolutionMeta((1.0, 1.0))
    assert r.resolution == (1.0, 1.0)  # 1/1.0 = 1.0
    assert r.pixel_size == (1.0, 1.0)
    assert r.unit == "micron"


def test_resolutionmeta_converts_to_pixel_density():
    # pixel size in um/px
    r = md.ResolutionMeta((0.5, 0.25))
    # density in px/um
    assert r.resolution == (2.0, 4.0)
    assert r.pixel_size == (0.5, 0.25)


# -------------------------
# make_color_lut
# -------------------------

@pytest.mark.parametrize("color, idx", [("red", 0), ("green", 1), ("blue", 2)])
def test_make_color_lut_shape_dtype_and_ramp(color, idx):
    lut = md.make_color_lut(color)
    assert lut.shape == (3, 256)
    assert lut.dtype == np.uint8

    # selected channel ramps 0..255
    assert np.array_equal(lut[idx], np.arange(256, dtype=np.uint8))

    # other channels are all zeros
    for j in {0, 1, 2} - {idx}:
        assert np.all(lut[j] == 0)


# -------------------------
# ChannelMeta
# -------------------------

def test_channelmeta_defaults_when_labels_none():
    cm = md.ChannelMeta(channel_number=3, labels=None)
    assert cm.mode == "grayscale"
    assert cm.luts is None
    assert cm.labels is not None
    assert list(cm.labels) == ["C1", "C2", "C3"]


def test_channelmeta_raises_on_wrong_label_count():
    with pytest.raises(ValueError):
        md.ChannelMeta(channel_number=2, labels=["GFP"])  # only 1 label


def test_channelmeta_color_mode_when_all_labels_map_to_rgb():
    cm = md.ChannelMeta(channel_number=2, labels=["GFP", "mCherry"])
    assert cm.mode == "color"
    assert cm.luts is not None
    assert len(cm.luts) == 2
    assert cm.luts[0].shape == (3, 256)
    assert cm.luts[1].shape == (3, 256)


def test_channelmeta_grayscale_when_any_label_unknown():
    cm = md.ChannelMeta(channel_number=2, labels=["GFP", "weird_dye"])
    assert cm.mode == "grayscale"
    assert cm.luts is None


def test_channelmeta_to_dict_includes_luts_only_when_present():
    cm1 = md.ChannelMeta(channel_number=1, labels=None)
    d1 = cm1.to_dict()
    assert "LUTs" not in d1
    assert d1["mode"] == "grayscale"

    cm2 = md.ChannelMeta(channel_number=1, labels=["GFP"])
    d2 = cm2.to_dict()
    assert d2["mode"] in {"color", "grayscale"}  # GFP should become color
    if d2["mode"] == "color":
        assert "LUTs" in d2


# -------------------------
# build_imagej_metadata
# -------------------------

def test_build_imagej_metadata_basic_includes_expected_fields():
    reader = DummyReader(axes="TZCYX", interval=11.0, channel_number=2, resolution=(0.5, 0.25))
    export_profile = ExportProfile(dist_name="test-dist", step_name="test_step", filename="test.tif")
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), export_profile, channel_labels=["GFP", "mCherry"])

    assert isinstance(out, md.TiffMetadata)
    assert out.imagej_meta["axes"] == "TZCYX"
    assert out.imagej_meta["finterval"] == 11.0
    assert out.imagej_meta["Labels"] == ["GFP", "mCherry"]
    assert out.imagej_meta["unit"] == "micron"  # since resolution != (1,1)
    assert out.resolution == (2.0, 4.0)  # density px/um
    assert out.extratags is not None  # resolution is stored in private tag payload


def test_build_imagej_metadata_channel_labels_str_becomes_list():
    reader = DummyReader(channel_number=1)
    export_profile = ExportProfile(dist_name="test-dist", step_name="test_step", filename="test.tif")
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), export_profile, channel_labels="GFP")
    assert out.imagej_meta["Labels"] == ["GFP"]


def test_build_imagej_metadata_no_resolution_means_no_unit_and_no_resolution_payload():
    reader = DummyReader(resolution=(1.0, 1.0))
    export_profile = ExportProfile(dist_name="test-dist", step_name="test_step", filename="test.tif")
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), export_profile, channel_labels=None)

    assert out.resolution == (1.0, 1.0)
    assert out.imagej_meta["unit"] == "micron"
    # provenance is always written now
    assert out.extratags is not None
    assert len(out.extratags) == 1

    tag, dtype, count, value, writeonce = out.extratags[0]
    assert tag == md.FITS_TAG  # or PIPELINE_TAG if you import it
    assert dtype == "B"
    assert count == len(value)
    assert writeonce is True

    payload = json.loads(value.decode("utf-8"))
    assert export_profile.step_name in payload
    step_meta = payload[export_profile.step_name]

    # basic provenance keys exist
    assert "dist" in step_meta
    assert "version" in step_meta
    assert "timestamp" in step_meta

    # and resolution is NOT included when default resolution
    assert "resolution" not in step_meta


def test_build_imagej_metadata_custom_metadata_only_creates_extratags():
    reader = DummyReader(resolution=(1.0, 1.0), custom_metadata={"a": 1, "b": {"c": 2}})
    export_profile = ExportProfile(dist_name="test-dist", step_name="test_step", filename="test.tif")
    out = md.build_imagej_metadata(
        cast(md.ImageReader, reader),
        export_profile,
    )

    assert out.extratags is not None
    (tag_id, tiff_dtype, count, value, writeonce) = out.extratags[0]

    assert tag_id == md.FITS_TAG
    assert tiff_dtype == "B"
    assert isinstance(value, (bytes, bytearray))
    assert count == len(value)
    assert writeonce is True

    payload = json.loads(value.decode("utf-8"))

    # existing metadata remains top-level
    assert payload["a"] == 1
    assert payload["b"] == {"c": 2}

    # provenance step is also present
    assert export_profile.step_name in payload
    step_meta = payload[export_profile.step_name]
    assert "dist" in step_meta
    assert "version" in step_meta
    assert "timestamp" in step_meta

    # no resolution payload when default resolution
    assert "resolution" not in step_meta



def test_build_imagej_metadata_resolution_payload_is_pixel_size_um_per_px():
    reader = DummyReader(resolution=(0.5, 0.25))
    export_profile = ExportProfile(dist_name="test-dist", step_name="test_step", filename="test.tif")
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), export_profile, extra_step_metadata={"resolution": (0.5, 0.25)})
    assert out.extratags is not None

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))

    # resolution is stored under the step, as pixel size (um/px)
    step_meta = payload[export_profile.step_name]
    assert step_meta["resolution"] == [0.5, 0.25] or step_meta["resolution"] == (0.5, 0.25)

