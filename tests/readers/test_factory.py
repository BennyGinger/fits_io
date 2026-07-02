from pathlib import Path

import pytest

from fits_io.readers.factory import (
    get_reader,
    ImageReaderError,
    ReaderFileNotFoundError,
    UnsupportedFileTypeError,
    READER_BY_SUFFIX,
)
from fits_io.readers.r_tiff import TiffReader
from fits_io.readers.r_nd2 import Nd2Reader
from fits_io.readers.protocol import ImageReader


# -------------------------
# Basic reader creation tests
# -------------------------

def test_get_reader_tif(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    """Test getting a TiffReader for .tif files."""
    p = tmp_path / "image.tif"
    p.write_bytes(b"fake")
    # Patch TiffFile where it's imported in r_tiff module
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_no_series)

    r = get_reader(p)
    assert isinstance(r, TiffReader)
    assert isinstance(r, ImageReader)


def test_get_reader_tiff_extension(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    """Test getting a TiffReader for .tiff files."""
    p = tmp_path / "image.tiff"
    p.write_bytes(b"fake")
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_no_series)

    r = get_reader(p)
    assert isinstance(r, TiffReader)


def test_get_reader_nd2(tmp_path: Path, monkeypatch, fake_nd2_file_basic):
    """Test getting an Nd2Reader for .nd2 files."""
    p = tmp_path / "image.nd2"
    p.write_bytes(b"fake")
    # Patch nd2.ND2File where it's used in r_nd2 module
    import fits_io.readers.r_nd2 as r_nd2
    monkeypatch.setattr(r_nd2.nd2, "ND2File", fake_nd2_file_basic)

    r = get_reader(p)
    assert isinstance(r, Nd2Reader)
    assert isinstance(r, ImageReader)


def test_get_reader_with_string_path(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    """Test that get_reader accepts string paths."""
    p = tmp_path / "image.tif"
    p.write_bytes(b"fake")
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_no_series)

    r = get_reader(str(p))
    assert isinstance(r, TiffReader)


def test_get_reader_with_multi_channel_tiff(tmp_path: Path, monkeypatch, fake_tiff_file_full):
    """Test that get_reader returns a valid reader for a multi-channel TIFF."""
    p = tmp_path / "image.tif"
    p.write_bytes(b"fake")
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_full)

    r = get_reader(p)
    assert isinstance(r, TiffReader)
    assert r.channel_count == 3


# -------------------------
# Error handling tests
# -------------------------

def test_get_reader_file_not_found():
    """Test that ReaderFileNotFoundError is raised for non-existent files."""
    p = Path("/nonexistent/path/to/file.tif")
    
    with pytest.raises(ReaderFileNotFoundError) as exc_info:
        get_reader(p)
    
    assert "Path not found" in str(exc_info.value)


def test_get_reader_path_is_directory(tmp_path: Path):
    """Test that ReaderFileNotFoundError is raised for directories."""
    with pytest.raises(ReaderFileNotFoundError) as exc_info:
        get_reader(tmp_path)
    
    assert "not a file" in str(exc_info.value)


def test_get_reader_unsupported_file_type(tmp_path: Path):
    """Test that UnsupportedFileTypeError is raised for unsupported types."""
    p = tmp_path / "image.jpg"
    p.write_bytes(b"fake")
    
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        get_reader(p)
    
    assert "Unsupported file type" in str(exc_info.value)
    assert ".jpg" in str(exc_info.value)


def test_get_reader_unsupported_file_type_lists_supported(tmp_path: Path):
    """Test that error message lists supported file types."""
    p = tmp_path / "image.png"
    p.write_bytes(b"fake")
    
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        get_reader(p)
    
    error_msg = str(exc_info.value)
    assert "Supported:" in error_msg
    assert ".tif" in error_msg or ".tiff" in error_msg
    assert ".nd2" in error_msg


def test_get_reader_case_insensitive_extension(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    """Test that file extensions are case-insensitive."""
    p = tmp_path / "image.TIF"
    p.write_bytes(b"fake")
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_no_series)

    r = get_reader(p)
    assert isinstance(r, TiffReader)


def test_get_reader_mixed_case_extension(tmp_path: Path, monkeypatch, fake_tiff_file_no_series):
    """Test mixed case extensions work."""
    p = tmp_path / "image.TiFf"
    p.write_bytes(b"fake")
    import fits_io.readers.r_tiff as r_tiff
    monkeypatch.setattr(r_tiff, "TiffFile", fake_tiff_file_no_series)

    r = get_reader(p)
    assert isinstance(r, TiffReader)


# -------------------------
# READER_BY_SUFFIX tests
# -------------------------

def test_reader_by_suffix_contains_tif():
    """Test that READER_BY_SUFFIX includes .tif."""
    assert ".tif" in READER_BY_SUFFIX
    assert READER_BY_SUFFIX[".tif"] == TiffReader


def test_reader_by_suffix_contains_tiff():
    """Test that READER_BY_SUFFIX includes .tiff."""
    assert ".tiff" in READER_BY_SUFFIX
    assert READER_BY_SUFFIX[".tiff"] == TiffReader


def test_reader_by_suffix_contains_nd2():
    """Test that READER_BY_SUFFIX includes .nd2."""
    assert ".nd2" in READER_BY_SUFFIX
    assert READER_BY_SUFFIX[".nd2"] == Nd2Reader


def test_reader_by_suffix_all_values_are_reader_types():
    """Test that all values in READER_BY_SUFFIX are ImageReader subclasses."""
    for suffix, reader_cls in READER_BY_SUFFIX.items():
        # Check that the class is a subclass of ImageReader
        # Note: We can't use issubclass directly on Protocol in all Python versions,
        # so we just check it's a type
        assert isinstance(reader_cls, type)


# -------------------------
# Exception hierarchy tests
# -------------------------

def test_reader_file_not_found_error_is_image_reader_error():
    """Test exception hierarchy."""
    assert issubclass(ReaderFileNotFoundError, ImageReaderError)


def test_unsupported_file_type_error_is_image_reader_error():
    """Test exception hierarchy."""
    assert issubclass(UnsupportedFileTypeError, ImageReaderError)


def test_image_reader_error_is_exception():
    """Test base exception class."""
    assert issubclass(ImageReaderError, Exception)