# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
from numpy.typing import NDArray
import numpy as np

from fits_io.image_reader import ImageReader, StatusFlag
import pytest

# Adjust this import to your actual module path
# e.g. from fits_io.writer import generate_convert_save_paths, build_output_path
from fits_io.filesystem import get_save_dirs, build_output_path, mkdirs_paths
from fits_io.writer import _get_array_to_export


class DummyReader(ImageReader):
    img_path: Path

    def __init__(self, img_path: Path, series_number: int, channel_labels: list[str]):
        self.img_path = img_path
        self._series_number = series_number
        self._channel_labels = channel_labels

    @property
    def series_number(self) -> int:
        return self._series_number

    # --- required stubs ---
    @classmethod
    def can_read(cls, path: Path) -> bool: ...
    @property
    def axes(self) -> list[str]:
        return ["CYX"]
    @property
    def compression_method(self) -> str | None:
        return None
    @property
    def status(self) -> StatusFlag:
        return "active"
    @property
    def export_status(self) -> str:
        return "fits_io.status: active\n"
    @property
    def channel_number(self) -> list[int]:
        return [len(self._channel_labels)]
    @property
    def channel_labels(self) -> list[str] | None:
        return self._channel_labels
    def axis_index(self, axis) -> list[int | None]:
        return [None]
    @property
    def resolution(self) -> list[tuple[float, float] | None]:
        return [(1.0, 1.0)]
    @property
    def interval(self) -> float | None:
        return None
    @property
    def custom_metadata(self) -> dict:
        return {}

    # --- behavior we want to test ---
    def get_array(self, z_projection=None) -> NDArray:
        # pretend: returns all channels
        return np.zeros((len(self._channel_labels), 10, 10), dtype=np.uint8)

    def get_channel(self, channel, z_projection=None) -> NDArray:
        # channel may be str or list[str]
        if isinstance(channel, list):
            return np.zeros((len(channel), 10, 10), dtype=np.uint8)
        return np.zeros((1, 10, 10), dtype=np.uint8)


def test_build_output_path_joins_series_dir_and_save_name(tmp_path: Path) -> None:
    series_dir = tmp_path / "sample_s1"
    out = build_output_path(series_dir, save_name="fits_masks.tif")
    assert out == series_dir / "fits_masks.tif"

@pytest.mark.parametrize("bad_save_name", ["", None, 123])
def test_build_output_path_rejects_bad_save_name(tmp_path: Path, bad_save_name) -> None:
    series_dir = tmp_path / "sample_s1"
    with pytest.raises(ValueError):
        build_output_path(series_dir, save_name=bad_save_name)  # type: ignore[arg-type]
        
def test_generate_save_dirs_creates_s1_for_single_series(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "my_image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    reader = DummyReader(img_path=input_path, series_number=1, channel_labels=["C1"])

    dirs = get_save_dirs(reader)
    
    # Convert to list if single Path
    if isinstance(dirs, Path):
        dirs = [dirs]
    
    # Create the directories
    dirs = mkdirs_paths(dirs)
    
    assert len(dirs) == 1
    expected_dir = input_path.parent / "my_image_s1"
    assert dirs[0] == expected_dir
    assert expected_dir.exists()
    assert expected_dir.is_dir()

    # if you want the file path, that’s a separate step:
    out = build_output_path(expected_dir, save_name="fits_array.tif")
    assert out == expected_dir / "fits_array.tif"


def test_generate_save_dirs_creates_one_folder_per_series(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "sample.nd2"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    reader = DummyReader(img_path=input_path, series_number=3, channel_labels=["C1", "C2"])

    dirs = get_save_dirs(reader)
    
    # Convert to list if single Path
    if isinstance(dirs, Path):
        dirs = [dirs]
    
    # Create the directories
    dirs = mkdirs_paths(dirs)

    assert len(dirs) == 3
    assert dirs == [
        input_path.parent / "sample_s1",
        input_path.parent / "sample_s2",
        input_path.parent / "sample_s3",
    ]
    for d in dirs:
        assert d.exists()
        assert d.is_dir()



def test_generate_save_dirs_is_idempotent_on_existing_dirs(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    # Pre-create dirs
    (input_path.parent / "image_s1").mkdir(parents=True, exist_ok=True)
    (input_path.parent / "image_s2").mkdir(parents=True, exist_ok=True)

    reader = DummyReader(img_path=input_path, series_number=2, channel_labels=["C1", "C2"])

    dirs = get_save_dirs(reader)

    assert dirs == [
        input_path.parent / "image_s1",
        input_path.parent / "image_s2",
    ]

# --- Tests for _get_array_to_export ---
def test_get_array_to_export_all_channels(tmp_path: Path) -> None:
    reader = DummyReader(
        img_path=tmp_path / "img.tif",
        series_number=1,
        channel_labels=["GFP", "RFP"],
    )

    arrays, used_channels = _get_array_to_export(
        reader,
        channel_labels=["GFP", "RFP"],
        export_channels="all",
    )

    assert used_channels == ["GFP", "RFP"]
    assert len(arrays) == 1
    assert arrays[0].shape == (2, 10, 10)
    
def test_get_array_to_export_single_channel(tmp_path: Path) -> None:
    reader = DummyReader(
        img_path=tmp_path / "img.tif",
        series_number=1,
        channel_labels=["GFP", "RFP"],
    )

    arrays, used_channels = _get_array_to_export(
        reader,
        channel_labels=["GFP", "RFP"],
        export_channels="GFP",
    )

    assert used_channels == ["GFP"]
    assert len(arrays) == 1
    assert arrays[0].shape == (1, 10, 10)
    
def test_get_array_to_export_multiple_channels(tmp_path: Path) -> None:
    reader = DummyReader(
        img_path=tmp_path / "img.tif",
        series_number=1,
        channel_labels=["GFP", "RFP", "DAPI"],
    )

    arrays, used_channels = _get_array_to_export(
        reader,
        channel_labels=["GFP", "RFP", "DAPI"],
        export_channels=["RFP", "DAPI"],
    )

    assert used_channels == ["RFP", "DAPI"]
    assert arrays[0].shape == (2, 10, 10)
    
def test_get_array_to_export_invalid_channel_falls_back_to_all(tmp_path: Path) -> None:
    reader = DummyReader(
        img_path=tmp_path / "img.tif",
        series_number=1,
        channel_labels=["GFP", "RFP"],
    )

    arrays, used_channels = _get_array_to_export(
        reader,
        channel_labels=["GFP", "RFP"],
        export_channels=["BAD"],
    )

    assert used_channels == ["GFP", "RFP"]
    assert arrays[0].shape == (2, 10, 10)
    
def test_get_array_to_export_channel_labels_as_string(tmp_path: Path) -> None:
    reader = DummyReader(
        img_path=tmp_path / "img.tif",
        series_number=1,
        channel_labels=["GFP"],
    )

    arrays, used_channels = _get_array_to_export(
        reader,
        channel_labels="GFP",
        export_channels="GFP",
    )

    assert used_channels == ["GFP"]
    assert arrays[0].shape == (1, 10, 10)