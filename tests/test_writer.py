# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
from numpy.typing import NDArray

from fits_io.image_reader import ImageReader
import pytest

# Adjust this import to your actual module path
# e.g. from fits_io.writer import generate_convert_save_paths, build_output_path
from fits_io.writer import generate_convert_save_paths, build_output_path


class DummyReader(ImageReader):
    img_path: Path
    @property
    def series_number(self) -> int:
        return self._series_number

    def __init__(self, img_path: Path, series_number: int):
        self.img_path = img_path
        self._series_number = series_number

    # minimal stubs to satisfy the ABC
    @classmethod
    def can_read(cls, path: Path) -> bool: ...
    @property
    def axes(self) -> str: ...
    @property
    def channel_number(self) -> int: ...
    @property
    def channel_labels(self) -> list[str] | None: ...
    @property
    def serie_axis_index(self) -> int | None: ...
    @property
    def resolution(self) -> tuple[float, float]: ...
    @property
    def interval(self) -> float | None: ...
    @property
    def custom_metadata(self) -> dict: ...
    def get_array(self) -> "NDArray | list[NDArray]": ...
    def get_channel(self, channel) -> NDArray: ...


def test_build_output_path_joins_series_dir_and_save_name(tmp_path: Path) -> None:
    series_dir = tmp_path / "sample_s1"
    out = build_output_path(series_dir, save_name="fits_masks.tif")
    assert out == series_dir / "fits_masks.tif"


def test_generate_convert_save_paths_creates_s1_for_single_series(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "my_image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")  # file doesn't need to be valid; only path is used

    reader = DummyReader(img_path=input_path, series_number=1)

    paths = generate_convert_save_paths(reader, save_name="fits_array.tif")

    assert len(paths) == 1
    expected_dir = input_path.parent / "my_image_s1"
    expected_path = expected_dir / "fits_array.tif"
    assert paths[0] == expected_path

    # Ensure folder is created by the function
    assert expected_dir.exists()
    assert expected_dir.is_dir()


def test_generate_convert_save_paths_creates_one_folder_per_series(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "sample.nd2"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    reader = DummyReader(img_path=input_path, series_number=3)

    paths = generate_convert_save_paths(reader, save_name="fits_array.tif")

    assert len(paths) == 3
    for i in range(1, 4):
        series_dir = input_path.parent / f"sample_s{i}"
        out_path = series_dir / "fits_array.tif"
        assert out_path in paths
        assert series_dir.exists()
        assert series_dir.is_dir()


@pytest.mark.parametrize("bad_series_number", [0, -1, -10])
def test_generate_convert_save_paths_raises_if_no_arrays(
    tmp_path: Path, bad_series_number: int
) -> None:
    input_path = tmp_path / "img.tif"
    input_path.write_bytes(b"")

    reader = DummyReader(img_path=input_path, series_number=bad_series_number)

    with pytest.raises(ValueError, match="no readable arrays"):
        generate_convert_save_paths(reader, save_name="fits_array.tif")


def test_generate_convert_save_paths_is_idempotent_on_existing_dirs(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    # Pre-create the target folders to ensure exist_ok=True behavior
    (input_path.parent / "image_s1").mkdir(parents=True, exist_ok=True)
    (input_path.parent / "image_s2").mkdir(parents=True, exist_ok=True)

    reader = DummyReader(img_path=input_path, series_number=2)

    paths = generate_convert_save_paths(reader, save_name="fits_array.tif")

    assert paths == [
        input_path.parent / "image_s1" / "fits_array.tif",
        input_path.parent / "image_s2" / "fits_array.tif",
    ]
