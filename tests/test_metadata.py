import json
from typing import cast

import numpy as np
import pytest

# adjust import if your package path is different
import fits_io.metadata as md


class DummyReader:
    """Minimal ImageReader-like object for build_imagej_metadata tests."""
    def __init__(self, axes="TZCYX", interval=11.0, channel_number=2, resolution=(0.5, 0.25)):
        self.axes = axes
        self.interval = interval
        self.channel_number = channel_number
        self.resolution = resolution


# -------------------------
# StackMeta
# -------------------------

def test_stackmeta_to_dict_with_interval():
    s = md.StackMeta(axes="TZCYX", finterval=11.0)
    d = s.to_dict()
    assert d == {"axes": "TZCYX", "finterval": 11.0}


def test_stackmeta_to_dict_without_interval():
    s = md.StackMeta(axes="YX", finterval=None)
    d = s.to_dict()
    assert d == {"axes": "YX"}
    assert "finterval" not in d


# -------------------------
# ResolutionMeta
# -------------------------

def test_resolutionmeta_default_is_none_density():
    r = md.ResolutionMeta((1.0, 1.0))
    assert r.resolution is None
    assert r.pixel_size == (1.0, 1.0)
    assert r.unit == "um"


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
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), channel_labels=["GFP", "mCherry"], custom_metadata=None)

    assert isinstance(out, md.TiffMetadata)
    assert out.imagej_meta["axes"] == "TZCYX"
    assert out.imagej_meta["finterval"] == 11.0
    assert out.imagej_meta["Labels"] == ["GFP", "mCherry"]
    assert out.imagej_meta["unit"] == "um"  # since resolution != (1,1)
    assert out.resolution == (2.0, 4.0)  # density px/um
    assert out.extratags is not None  # resolution is stored in private tag payload


def test_build_imagej_metadata_channel_labels_str_becomes_list():
    reader = DummyReader(channel_number=1)
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), channel_labels="GFP")
    assert out.imagej_meta["Labels"] == ["GFP"]


def test_build_imagej_metadata_no_resolution_means_no_unit_and_no_resolution_payload():
    reader = DummyReader(resolution=(1.0, 1.0))
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), channel_labels=None, custom_metadata=None)

    assert out.resolution is None
    assert "unit" not in out.imagej_meta
    # payload empty => extratags None
    assert out.extratags is None


def test_build_imagej_metadata_custom_metadata_only_creates_extratags():
    reader = DummyReader(resolution=(1.0, 1.0))
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), custom_metadata={"a": 1, "b": {"c": 2}})

    assert out.extratags is not None
    (tag_id, tiff_dtype, count, value, writeonce) = out.extratags[0]
    assert tag_id == md.PIPELINE_TAG
    assert tiff_dtype == "s"
    assert count == 0
    assert writeonce is True

    payload = json.loads(value)
    assert payload == {"a": 1, "b": {"c": 2}}
    assert "resolution" not in payload


def test_build_imagej_metadata_resolution_payload_is_pixel_size_um_per_px():
    reader = DummyReader(resolution=(0.5, 0.25))
    out = md.build_imagej_metadata(cast(md.ImageReader, reader), custom_metadata={"x": "y"})
    assert out.extratags is not None

    payload = json.loads(out.extratags[0][3])
    # stored payload uses pixel_size (um/px), not density
    assert payload["resolution"] == [0.5, 0.25] or payload["resolution"] == (0.5, 0.25)
    assert payload["x"] == "y"
