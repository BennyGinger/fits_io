# tests/test_metadata.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pytest

import fits_io.metadata as md


# ------------------------------------------------------------
# Shared test stubs / factories
# ------------------------------------------------------------

@dataclass
class ReaderStub:
    """
    Minimal ImageReader-like object for build_imagej_metadata tests.

    We keep this local (instead of using DummyReader from conftest) because
    metadata tests need to vary axes/interval/resolution/channel_number easily.
    """
    axes: list[str] = field(default_factory=lambda: ["TZCYX"])
    interval: float | None = 11.0
    channel_number: list[int] = field(default_factory=lambda: [2])
    resolution: list[tuple[float, float] | None] = field(default_factory=lambda: [(0.5, 0.25)])
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)

    export_status: str = "fits_io.status: active\n"
    status: str = "active"

    # Only needed if your metadata builder reads it (some versions did)
    channel_labels: None = None


@pytest.fixture
def reader_factory():
    """Factory to build ReaderStub with overrides, keeping tests concise."""
    def _make(**overrides: Any) -> ReaderStub:
        return ReaderStub(**overrides)
    return _make


@pytest.fixture
def build_meta():
    """Convenience wrapper for md.build_imagej_metadata with common args."""
    def _build(reader: ReaderStub, **kwargs: Any) -> md.TiffMetadata:
        return md.build_imagej_metadata(
            cast(md.ImageReader, reader),
            user_name=kwargs.pop("user_name", "test_user"),
            distribution=kwargs.pop("distribution", "test-dist"),
            step_name=kwargs.pop("step_name", "test_step"),
            **kwargs,
        )
    return _build


# -------------------------
# StackMeta
# -------------------------

def test_stackmeta_to_dict_with_interval():
    s = md.StackMeta(axes="TZCYX", status="active", user_name="unknown", finterval=11.0)
    assert s.to_dict() == {
        "axes": "TZCYX",
        "Info": "fits_io.status: active\nfits_io.user: unknown\n",
        "finterval": 11.0,
    }


def test_stackmeta_to_dict_without_interval():
    s = md.StackMeta(axes="YX", status="active", user_name="unknown", finterval=None)
    d = s.to_dict()
    assert d == {
        "axes": "YX",
        "Info": "fits_io.status: active\nfits_io.user: unknown\n",
    }
    assert "finterval" not in d


# -------------------------
# ResolutionMeta
# -------------------------

def test_resolutionmeta_default_is_identity():
    r = md.ResolutionMeta((1.0, 1.0))
    assert r.resolution == (1.0, 1.0)  # px/um density
    assert r.pixel_size == (1.0, 1.0)  # um/px
    assert r.unit == "micron"


def test_resolutionmeta_converts_to_pixel_density():
    r = md.ResolutionMeta((0.5, 0.25))  # um/px
    assert r.resolution == (2.0, 4.0)   # px/um
    assert r.pixel_size == (0.5, 0.25)


# -------------------------
# make_color_lut
# -------------------------

@pytest.mark.parametrize("color, idx", [("red", 0), ("green", 1), ("blue", 2)])
def test_make_color_lut_shape_dtype_and_ramp(color: str, idx: int):
    lut = md.make_color_lut(color)
    assert lut.shape == (3, 256)
    assert lut.dtype == np.uint8
    assert np.array_equal(lut[idx], np.arange(256, dtype=np.uint8))
    for j in {0, 1, 2} - {idx}:
        assert np.all(lut[j] == 0)


# -------------------------
# ChannelMeta
# -------------------------

def test_channelmeta_defaults_when_labels_none():
    cm = md.ChannelMeta(channel_number=3, labels=None)
    assert cm.mode == "grayscale"
    assert cm.luts is None
    assert cm.labels is None


def test_channelmeta_raises_on_wrong_label_count():
    with pytest.raises(ValueError):
        md.ChannelMeta(channel_number=2, labels=["GFP"])


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
    assert d1["mode"] == "grayscale"
    assert "LUTs" not in d1

    cm2 = md.ChannelMeta(channel_number=1, labels=["GFP"])
    d2 = cm2.to_dict()
    if d2["mode"] == "color":
        assert "LUTs" in d2


# -------------------------
# build_imagej_metadata
# -------------------------

def test_build_imagej_metadata_basic_includes_expected_fields(reader_factory, build_meta):
    reader = reader_factory(
        axes=["TZCYX"],
        interval=11.0,
        channel_number=[2],
        resolution=[(0.5, 0.25)],
    )
    out = build_meta(reader, channel_labels=["GFP", "mCherry"])

    assert isinstance(out, md.TiffMetadata)
    assert out.imagej_meta["axes"] == "TZCYX"
    assert out.imagej_meta["finterval"] == 11.0
    assert out.imagej_meta["Labels"] == ["GFP", "mCherry"]
    assert out.imagej_meta["unit"] == "micron"
    assert out.resolution == (2.0, 4.0)
    assert out.extratags is not None


def test_build_imagej_metadata_channel_labels_str_becomes_list(reader_factory, build_meta):
    reader = reader_factory(channel_number=[1], resolution=[(1.0, 1.0)])
    out = build_meta(reader, channel_labels="GFP")
    assert out.imagej_meta["Labels"] == ["GFP"]


def test_build_imagej_metadata_default_resolution_still_writes_provenance(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)])
    out = build_meta(reader, channel_labels=None)

    assert out.resolution == (1.0, 1.0)
    assert out.extratags is not None
    assert len(out.extratags) == 1

    tag, dtype, count, value, writeonce = out.extratags[0]
    assert tag == md.FITS_TAG
    assert dtype == "B"
    assert count == len(value)
    assert writeonce is True

    payload = json.loads(value.decode("utf-8"))
    assert "test_step" in payload
    step_meta = payload["test_step"]

    assert "dist" in step_meta
    assert "version" in step_meta
    assert "timestamp" in step_meta

    # Default resolution should not be duplicated into step payload
    assert "resolution" not in step_meta


def test_build_imagej_metadata_custom_metadata_preserved(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"a": 1, "b": {"c": 2}})
    out = build_meta(reader)

    assert out.extratags is not None
    (_, _, _, raw, _) = out.extratags[0]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["a"] == 1
    assert payload["b"] == {"c": 2}
    assert "test_step" in payload


def test_build_imagej_metadata_resolution_payload_is_pixel_size_um_per_px(reader_factory, build_meta):
    reader = reader_factory(resolution=[(0.5, 0.25)])
    out = build_meta(reader, extra_step_metadata={"resolution": (0.5, 0.25)})

    assert out.extratags is not None
    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["test_step"]["resolution"] == [0.5, 0.25]


def test_build_imagej_metadata_with_new_status_preserves_custom_metadata(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(reader, new_status="skip")

    assert "skip" in out.imagej_meta["Info"]

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["existing"] == "data"
    assert "test_step" in payload


def test_build_imagej_metadata_add_provenance_false_skips_provenance_step(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(reader, add_provenance=False)

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["existing"] == "data"
    assert "test_step" not in payload


def test_build_imagej_metadata_add_provenance_false_ignores_extra_step_metadata(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(
        reader,
        extra_step_metadata={"resolution": (0.5, 0.25)},
        add_provenance=False,
    )

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["existing"] == "data"
    assert "test_step" not in payload


# -------------------------
# _normalize_channel_labels
# -------------------------

def test_normalize_channel_labels_none_returns_none():
    assert md._validate_labels(None, n_channels=3) is None


def test_normalize_channel_labels_string_with_one_channel():
    assert md._validate_labels("GFP", n_channels=1) == ["GFP"]


def test_normalize_channel_labels_string_with_multiple_channels_raises():
    with pytest.raises(ValueError):
        md._validate_labels("GFP", n_channels=2)


def test_normalize_channel_labels_sequence_matches_channel_count():
    labels = ["GFP", "mCherry"]
    assert md._validate_labels(labels, n_channels=2) == labels


def test_normalize_channel_labels_sequence_wrong_length_raises():
    with pytest.raises(ValueError):
        md._validate_labels(["GFP"], n_channels=2)
