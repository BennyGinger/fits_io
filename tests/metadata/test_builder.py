# tests/metadata/test_builder.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import pytest

from fits_io.metadata.builder import build_metadata
from fits_io.metadata.models import TiffMetadata
from fits_io.metadata.provenance import FITS_TAG
from fits_io.readers._types import StatusFlag
from fits_io.readers.protocol import ImageReader


# ------------------------------------------------------------
# Shared test stubs / factories
# ------------------------------------------------------------

@dataclass
class ReaderStub:
    """
    Minimal ImageReader-like object for build_metadata tests.

    We keep this local (instead of using DummyReader from conftest) because
    metadata tests need to vary axes/interval/resolution/channel_number easily.
    """
    axes: list[str] = field(default_factory=lambda: ["TZCYX"])
    interval: float | None = 11.0
    channel_number: list[int] = field(default_factory=lambda: [2])
    resolution: list[tuple[float, float] | None] = field(default_factory=lambda: [(0.5, 0.25)])
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)
    channel_labels: list[str] = field(default_factory=lambda: ["C_1", "C_2"])

    export_status: str = "fits_io.status: active\n"
    status: StatusFlag = "active"


@pytest.fixture
def reader_factory():
    """Factory to build ReaderStub with overrides, keeping tests concise."""
    def _make(**overrides: Any) -> ReaderStub:
        return ReaderStub(**overrides)
    return _make


@pytest.fixture
def build_meta():
    """Convenience wrapper for build_metadata with common args."""
    def _build(reader: ReaderStub, **kwargs: Any) -> TiffMetadata:
        return build_metadata(
            cast(ImageReader, reader),
            new_user=kwargs.pop("user_name", "test_user"),
            distribution=kwargs.pop("distribution", "test-dist"),
            step_name=kwargs.pop("step_name", "test_step"),
            **kwargs,
        )
    return _build


# -------------------------
# build_metadata
# -------------------------

def test_build_metadata_basic_includes_expected_fields(reader_factory, build_meta):
    reader = reader_factory(
        axes=["TZCYX"],
        interval=11.0,
        channel_number=[2],
        resolution=[(0.5, 0.25)],
        channel_labels=["GFP", "mCherry"],
    )
    out = build_meta(reader, channel_labels=["GFP", "mCherry"])

    assert isinstance(out, TiffMetadata)
    assert out.imagej_meta["axes"] == "TZCYX"
    assert out.imagej_meta["finterval"] == 11.0
    assert out.imagej_meta["Labels"] == ["GFP", "mCherry"]
    assert out.imagej_meta["unit"] == "micron"
    assert out.resolution == (2.0, 4.0)
    assert out.extratags is not None


def test_build_metadata_channel_labels_str_becomes_list(reader_factory, build_meta):
    reader = reader_factory(channel_number=[1], resolution=[(1.0, 1.0)], channel_labels=["GFP"])
    out = build_meta(reader, channel_labels="GFP")
    assert out.imagej_meta["Labels"] == ["GFP"]


def test_build_metadata_default_resolution_still_writes_provenance(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)])
    out = build_meta(reader, channel_labels=None)

    assert out.resolution == (1.0, 1.0)
    assert out.extratags is not None
    assert len(out.extratags) == 1

    tag, dtype, count, value, writeonce = out.extratags[0]
    assert tag == FITS_TAG
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


def test_build_metadata_custom_metadata_preserved(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"a": 1, "b": {"c": 2}})
    out = build_meta(reader)

    assert out.extratags is not None
    (_, _, _, raw, _) = out.extratags[0]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["a"] == 1
    assert payload["b"] == {"c": 2}
    assert "test_step" in payload


def test_build_metadata_resolution_payload_is_pixel_size_um_per_px(reader_factory, build_meta):
    reader = reader_factory(resolution=[(0.5, 0.25)])
    out = build_meta(reader, extra_step_metadata={"resolution": (0.5, 0.25)})

    assert out.extratags is not None
    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["test_step"]["resolution"] == [0.5, 0.25]


def test_build_metadata_with_new_status_preserves_custom_metadata(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(reader, new_status="skip")

    assert "skip" in out.imagej_meta["Info"]

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["existing"] == "data"
    assert "test_step" in payload


def test_build_metadata_add_step_meta_false_skips_provenance_step(reader_factory, build_meta):
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(reader, add_step_meta=False)

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["existing"] == "data"
    assert "test_step" not in payload


def test_build_metadata_add_step_meta_false_still_adds_extra_step_metadata(reader_factory, build_meta):
    """When add_step_meta=False, provenance profile is skipped but extra_step_metadata is still added."""
    reader = reader_factory(resolution=[(1.0, 1.0)], custom_metadata={"existing": "data"})
    out = build_meta(
        reader,
        extra_step_metadata={"resolution": (0.5, 0.25)},
        add_step_meta=False,
    )

    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))

    assert payload["existing"] == "data"
    # extra_step_metadata is still added even when add_step_meta=False
    assert "test_step" in payload
    assert payload["test_step"]["resolution"] == [0.5, 0.25]
    # But provenance profile (dist, version, timestamp) is not added
    assert "dist" not in payload["test_step"]
    assert "version" not in payload["test_step"]
    assert "timestamp" not in payload["test_step"]


def test_build_metadata_z_projection_drops_z_axis(reader_factory, build_meta):
    reader = reader_factory(axes=["TZCYX"])
    out = build_meta(reader, z_projection="max")
    
    assert "Z" not in out.imagej_meta["axes"]
    assert out.imagej_meta["axes"] == "TCYX"


def test_build_metadata_z_projection_in_step_metadata(reader_factory, build_meta):
    reader = reader_factory(axes=["TZCYX"])
    out = build_meta(reader, z_projection="mean", extra_step_metadata={"param": "value"})
    
    raw = out.extratags[0][3]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["z_projection_method"] == "mean"
    assert payload["test_step"]["param"] == "value"


def test_build_metadata_single_channel_drops_c_axis(reader_factory, build_meta):
    reader = reader_factory(axes=["TCYX"], channel_number=[3], channel_labels=["GFP"])
    out = build_meta(reader, channel_labels="GFP")
    
    assert "C" not in out.imagej_meta["axes"]
    assert out.imagej_meta["axes"] == "TYX"


def test_build_metadata_uses_reader_channel_labels_when_none_provided(reader_factory, build_meta):
    reader = reader_factory(channel_number=[3], channel_labels=["Red", "Green", "Blue"])
    out = build_meta(reader, channel_labels=None)
    
    assert out.imagej_meta["Labels"] == ["Red", "Green", "Blue"]


def test_build_metadata_series_index_selects_correct_metadata(reader_factory, build_meta):
    reader = reader_factory(
        axes=["CYX", "TCYX"],
        channel_number=[2, 3],
        channel_labels=["C_1", "C_2", "C_3"],  # provide labels for max channels
        resolution=[(0.5, 0.5), (0.3, 0.3)]
    )
    out = build_meta(reader, series_index=1)
    
    assert out.imagej_meta["axes"] == "TCYX"



