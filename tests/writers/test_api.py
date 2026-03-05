# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fits_io.writers import api




# -----------------------------------
# High-level: convert_to_fits_tif()
# -----------------------------------

def test_convert_to_fits_tif_returns_output_paths(writer_harness, dummy_reader) -> None:
    out = api.convert_to_fits_tif(dummy_reader)

    assert out == [writer_harness.save_dirs[0] / "fits.tif"]
    assert len(writer_harness.saved) == 1


def test_convert_to_fits_tif_writes_one_file_per_series(writer_harness, dummy_reader) -> None:
    s1 = writer_harness.tmp_path / "img_s1"
    s2 = writer_harness.tmp_path / "img_s2"
    writer_harness.save_dirs = [s1, s2]

    a1 = np.ones((3, 3), dtype=np.uint8)
    a2 = np.ones((3, 3), dtype=np.uint8) * 2
    writer_harness.arrays = [a1, a2]

    out_paths = api.convert_to_fits_tif(
        dummy_reader,
        output_name="fits.tif",
        compression="zlib",
    )

    assert out_paths == [s1 / "fits.tif", s2 / "fits.tif"]
    assert [c["path"] for c in writer_harness.saved] == [s1 / "fits.tif", s2 / "fits.tif"]
    assert [c["array"] for c in writer_harness.saved] == [a1, a2]
    assert [c["compression"] for c in writer_harness.saved] == ["zlib", "zlib"]

    # sanity: metadata built for both series_index values
    assert [kw.get("series_index") for kw in writer_harness.md_calls] == [0, 1]


# -----------------------------------
# save_fits_array()
# -----------------------------------

def test_save_fits_array_raises_on_multi_series(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dummy_reader) -> None:
    """Test that save_fits_array raises ValueError for multi-series."""
    # Make dummy_reader return list of arrays (multi-series)
    dummy_reader.array = [np.ones((3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
    
    from fits_io.metadata import builder
    monkeypatch.setattr(builder, "build_metadata", lambda *args, **kwargs: SimpleNamespace(imagej_meta={}, resolution=None, extratags=[]))
    
    with pytest.raises(ValueError, match="Multiple series detected"):
        api.export_source_array(dummy_reader)


# -----------------------------------
# set_status()
# -----------------------------------

def test_set_status_raises_on_non_tiff_reader(dummy_reader) -> None:
    """Test that set_status raises TypeError for non-TiffReader."""
    with pytest.raises(TypeError, match="only supports .tif/.tiff files"):
        api.set_status(dummy_reader, "skip")


def test_set_status_raises_on_invalid_status(writer_harness_tiff, dummy_reader) -> None:
    """Test that set_status raises ValueError for invalid status."""
    with pytest.raises(ValueError, match="Invalid status"):
        api.set_status(dummy_reader, "invalid_status")  # type: ignore[arg-type]


def test_set_status_raises_on_multi_series(writer_harness_tiff, dummy_reader) -> None:
    """Test that set_status raises ValueError for multi-series."""
    dummy_reader.array = [np.ones((3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
    
    with pytest.raises(ValueError, match="Expected a single array"):
        api.set_status(dummy_reader, "skip")


def test_set_status_sets_valid_status(writer_harness_tiff, dummy_reader) -> None:
    """Test that set_status successfully sets a valid status."""
    api.set_status(dummy_reader, "skip")
    
    # Check that save_tiff was called
    assert len(writer_harness_tiff.saved) == 1
    assert writer_harness_tiff.saved[0]["path"] == dummy_reader.img_path


# -----------------------------------
# set_channel_labels()
# -----------------------------------

def test_set_channel_labels_raises_on_non_tiff_reader(dummy_reader) -> None:
    """Test that set_channel_labels raises TypeError for non-TiffReader."""
    with pytest.raises(TypeError, match="only supports .tif/.tiff files"):
        api.set_channel_labels(dummy_reader, ["DAPI", "GFP"])


def test_set_channel_labels_raises_on_multi_series(writer_harness_tiff, dummy_reader) -> None:
    """Test that set_channel_labels raises ValueError for multi-series."""
    dummy_reader.array = [np.ones((3, 3), dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
    
    with pytest.raises(ValueError, match="Expected a single array"):
        api.set_channel_labels(dummy_reader, ["DAPI", "GFP"])


def test_set_channel_labels_sets_valid_labels(writer_harness_tiff, dummy_reader) -> None:
    """Test that set_channel_labels successfully sets labels."""
    api.set_channel_labels(dummy_reader, ["DAPI", "GFP", "RFP"])
    
    # Check that save_tiff was called
    assert len(writer_harness_tiff.saved) == 1
    assert writer_harness_tiff.saved[0]["path"] == dummy_reader.img_path
