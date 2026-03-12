import pytest

from fits_io.metadata.models import ChannelMeta, ResolutionMeta, StackMeta

# -------------------------
# StackMeta
# -------------------------

def test_stackmeta_to_dict_with_interval():
    s = StackMeta(axes="TZCYX", finterval=11.0)
    assert s.to_dict() == {
        "axes": "TZCYX",
        "finterval": 11.0,
    }


def test_stackmeta_to_dict_without_interval():
    s = StackMeta(axes="YX", finterval=None)
    d = s.to_dict()
    assert d == {
        "axes": "YX",
    }
    assert "finterval" not in d


# -------------------------
# ResolutionMeta
# -------------------------

def test_resolutionmeta_default_is_identity():
    r = ResolutionMeta((1.0, 1.0))
    assert r.resolution == (1.0, 1.0)  # px/um density
    assert r.pixel_size == (1.0, 1.0)  # um/px
    assert r.unit == "micron"


def test_resolutionmeta_converts_to_pixel_density():
    r = ResolutionMeta((0.5, 0.25))  # um/px
    assert r.resolution == (2.0, 4.0)   # px/um
    assert r.pixel_size == (0.5, 0.25)


# -------------------------
# ChannelMeta
# -------------------------

def test_channelmeta_defaults_when_labels_none():
    cm = ChannelMeta(channel_number=3, labels=None)
    assert cm.mode == "grayscale"
    assert cm.luts is None
    assert cm.labels is None


def test_channelmeta_raises_on_wrong_label_count():
    with pytest.raises(ValueError):
        ChannelMeta(channel_number=2, labels=["GFP"])


def test_channelmeta_color_mode_when_all_labels_map_to_rgb():
    cm = ChannelMeta(channel_number=2, labels=["GFP", "mCherry"])
    assert cm.mode == "color"
    assert cm.luts is not None
    assert len(cm.luts) == 2
    assert cm.luts[0].shape == (3, 256)
    assert cm.luts[1].shape == (3, 256)


def test_channelmeta_grayscale_when_any_label_unknown():
    cm = ChannelMeta(channel_number=2, labels=["GFP", "weird_dye"])
    assert cm.mode == "grayscale"
    assert cm.luts is None


def test_channelmeta_to_dict_includes_luts_only_when_present():
    cm1 = ChannelMeta(channel_number=1, labels=None)
    d1 = cm1.to_dict()
    assert d1["mode"] == "grayscale"
    assert "LUTs" not in d1

    cm2 = ChannelMeta(channel_number=1, labels=["GFP"])
    d2 = cm2.to_dict()
    if d2["mode"] == "color":
        assert "LUTs" in d2