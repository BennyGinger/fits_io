# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
import pytest

from fits_io.writers.filesystem import build_output_paths, get_save_dirs, mkdirs_paths, _ends_with_s_number

# -----------------------------
# Filesystem tests (unchanged)
# -----------------------------

def test_build_output_path_joins_series_dir_and_save_name(tmp_path: Path) -> None:
    from conftest import DummyReader

    reader = DummyReader(img_path=tmp_path / "sample.tif", series_idx=0)

    out = build_output_paths([reader], save_name="fits_masks.tif")
    # tmp_path name can already end with _sN in pytest temp dirs; current API then reuses parent.
    assert out == [(tmp_path / "fits_masks.tif")]


@pytest.mark.parametrize("bad_save_name", ["", None, 123])
def test_build_output_path_rejects_bad_save_name(tmp_path: Path, bad_save_name: object) -> None:
    from conftest import DummyReader

    reader = DummyReader(img_path=tmp_path / "sample.tif", series_idx=0)

    with pytest.raises(ValueError):
        build_output_paths([reader], save_name=bad_save_name)  # type: ignore[arg-type]


def test_get_save_dirs_creates_s1_for_single_series(tmp_path: Path) -> None:
    input_path = tmp_path / "input" / "my_image.tif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    from conftest import DummyReader

    reader = DummyReader(img_path=input_path, series_idx=0)
    dirs = get_save_dirs([reader])

    created = mkdirs_paths(dirs)
    assert len(created) == 1

    expected_dir = input_path.parent / "my_image_s1"
    assert created[0] == expected_dir
    assert expected_dir.exists() and expected_dir.is_dir()


def test_get_save_dirs_creates_one_folder_per_series(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "sample.nd2"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"")

    from conftest import DummyReader

    readers = [DummyReader(img_path=input_path, series_idx=i) for i in range(3)]
    dirs = get_save_dirs(readers)

    created = mkdirs_paths(dirs)
    assert created == [
        input_path.parent / "sample_s1",
        input_path.parent / "sample_s2",
        input_path.parent / "sample_s3",
    ]
    for d in created:
        assert d.exists() and d.is_dir()

# Test the regex helper
def test_ends_with_s_number_valid():
    """Test valid _sN patterns."""
    assert _ends_with_s_number("experiment_s1")
    assert _ends_with_s_number("experiment_s01")
    assert _ends_with_s_number("experiment_s123")
    assert _ends_with_s_number("my_data_s5")

def test_ends_with_s_number_invalid():
    """Test invalid patterns that should NOT match."""
    assert not _ends_with_s_number("experiment")
    assert not _ends_with_s_number("experiment_s")
    assert not _ends_with_s_number("experiment_s1234")  # Too many digits (max 3)
    assert not _ends_with_s_number("experiments1")  # No underscore
    assert not _ends_with_s_number("experiment_s1a")  # Has letter after
    assert not _ends_with_s_number("experiment_s1_extra")  # Has more after

# Test get_save_dirs with FITS file
def test_get_save_dirs_already_fits_file(tmp_path: Path):
    """When input is already in _sN directory, return parent as-is."""
    fits_dir = tmp_path / "experiment_s1"
    fits_dir.mkdir(parents=True)
    input_path = fits_dir / "array.tif"
    input_path.write_bytes(b"")
    
    from conftest import DummyReader
    
    dirs = get_save_dirs([DummyReader(img_path=input_path, series_idx=0)])
    assert dirs == [fits_dir]

# Test mkdirs_paths directly
def test_mkdirs_paths_single_path(tmp_path: Path):
    """Test creating directory from single Path."""
    target = tmp_path / "new_dir"
    result = mkdirs_paths(target)
    assert result == [target]
    assert target.exists() and target.is_dir()

def test_mkdirs_paths_list_of_paths(tmp_path: Path):
    """Test creating directories from list of Paths."""
    targets = [tmp_path / "dir1", tmp_path / "dir2", tmp_path / "dir3"]
    result = mkdirs_paths(targets)
    assert result == targets
    for t in targets:
        assert t.exists() and t.is_dir()

def test_mkdirs_paths_idempotent(tmp_path: Path):
    """Test that calling mkdirs_paths twice doesn't fail."""
    target = tmp_path / "existing_dir"
    mkdirs_paths(target)
    # Should not raise
    result = mkdirs_paths(target)
    assert result == [target]
    assert target.exists()

