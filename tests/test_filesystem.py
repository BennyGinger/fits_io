# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

import fits_io.writer as writer_module
from fits_io.filesystem import build_output_path, get_save_dirs, mkdirs_paths
from fits_io.image_reader import TiffReader
from fits_io.metadata import TiffMetadata


# -----------------------------
# Filesystem tests (unchanged)
# -----------------------------

def test_build_output_path_joins_series_dir_and_save_name(tmp_path: Path) -> None:
    series_dir = tmp_path / "sample_s1"
    out = build_output_path(series_dir, save_name="fits_masks.tif")
    assert out == series_dir / "fits_masks.tif"


@pytest.mark.parametrize("bad_save_name", ["", None, 123])
def test_build_output_path_rejects_bad_save_name(tmp_path: Path, bad_save_name: object) -> None:
    series_dir = tmp_path / "sample_s1"
    with pytest.raises(ValueError):
        build_output_path(series_dir, save_name=bad_save_name)  # type: ignore[arg-type]


def test_get_save_dirs_creates_s1_for_single_series(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "my_image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    # We only need img_path + series_number for get_save_dirs
    class _R:
        img_path = input_path
        series_number = 1

    dirs = get_save_dirs(_R())  # type: ignore[arg-type]
    if isinstance(dirs, Path):
        dirs = [dirs]

    created = mkdirs_paths(dirs)
    assert len(created) == 1

    expected_dir = input_path.parent / "my_image_s1"
    assert created[0] == expected_dir
    assert expected_dir.exists() and expected_dir.is_dir()


def test_get_save_dirs_creates_one_folder_per_series(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "sample.nd2"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    class _R:
        img_path = input_path
        series_number = 3

    dirs = get_save_dirs(_R())  # type: ignore[arg-type]
    if isinstance(dirs, Path):
        dirs = [dirs]

    created = mkdirs_paths(dirs)
    assert created == [
        input_path.parent / "sample_s1",
        input_path.parent / "sample_s2",
        input_path.parent / "sample_s3",
    ]
    for d in created:
        assert d.exists() and d.is_dir()


# --------------------------------
# set_channel_labels tests
# --------------------------------

def _make_dummy_tiff_reader(path: Path):
    """
    Create a TiffReader instance without triggering its real __post_init__/file IO.

    We only need:
      - img_path
      - get_array()
      - compression_method property
    """
    r = object.__new__(TiffReader)  # bypass dataclass init + __post_init__
    r.img_path = path
    return r


def test_set_channel_labels_rejects_non_tiff_reader(tmp_path: Path) -> None:
    class NotATiff:
        pass

    with pytest.raises(TypeError, match="only supports"):
        writer_module.set_channel_labels(NotATiff(), channel_labels=["C_1"])  # type: ignore[arg-type]


def test_set_channel_labels_raises_on_multi_series_array(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "img.tif"
    r = _make_dummy_tiff_reader(p)

    # Fake a multi-series result
    monkeypatch.setattr(r, "get_array", lambda *a, **k: [np.ones((2, 2), dtype=np.uint8)])  # type: ignore[attr-defined]
    monkeypatch.setattr(type(r), "compression_method", property(lambda self: "lzw"))  # type: ignore[arg-type]

    # Ensure metadata builder returns something valid
    monkeypatch.setattr(
        writer_module,
        "build_imagej_metadata",
        lambda *a, **k: TiffMetadata(imagej_meta={}, resolution=None, extratags=[]),
    )

    with pytest.raises(ValueError, match="multiple series"):
        writer_module.set_channel_labels(r, channel_labels=["C_1"])


def test_set_channel_labels_calls_save_with_same_path_and_existing_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "img.tif"
    r = _make_dummy_tiff_reader(p)

    arr = np.ones((3, 4), dtype=np.uint8)
