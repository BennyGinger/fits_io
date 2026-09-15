# -----------------------
# Low-level: save_tiff()
# -----------------------

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest

from fits_io.writers import core


def test_save_tiff_raises_on_empty_array(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {"n": 0}

    def fake_imwrite(*args: Any, **kwargs: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr(core, "imwrite", fake_imwrite)

    empty = np.array([], dtype=np.uint8)
    meta = SimpleNamespace(imagej_meta={}, resolution=None, extratags=[])

    with pytest.raises(ValueError, match="Cannot save empty array"):
        core.save_tiff(empty, tmp_path / "out.tif", meta)  # type: ignore[arg-type]

    assert called["n"] == 0


def test_save_tiff_preserves_interleaved_rgb_samples_axis(tmp_path: Path) -> None:
    """The final S axis describes RGB samples belonging to each pixel."""
    from tifffile import TiffFile

    array = np.zeros((2, 5, 6, 3), dtype=np.uint8)
    metadata = SimpleNamespace(
        imagej_meta={"axes": "TYXS"}, resolution=None, extratags=[])
    output = tmp_path / "rendered_tracking_display.tif"

    core.save_tiff(array, output, metadata, compression=None)  # type: ignore[arg-type]

    with TiffFile(output) as tiff:
        assert tiff.series[0].axes == "TYXS"
        assert tiff.series[0].shape == array.shape
        assert tiff.pages[0].photometric.name == "RGB"


@pytest.mark.parametrize(
    "compression, expected_predictor",
    [
        ("zlib", 2),
        ("deflate", 2),
        ("lzma", 2),
        ("lzw", None),
        (None, None),
    ],
)
def test_save_tiff_predictor_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compression: str | None,
    expected_predictor: int | None,
) -> None:
    captured: dict[str, Any] = {}

    def fake_imwrite(save_path: Path, img_array: np.ndarray, **kwargs: Any) -> None:
        captured["save_path"] = save_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(core, "imwrite", fake_imwrite)

    arr = np.ones((5, 6), dtype=np.uint16)
    meta = SimpleNamespace(imagej_meta={"axes": "YX"}, resolution=(1.0, 1.0), extratags=[])
    out_path = tmp_path / "out.tif"

    core.save_tiff(arr, out_path, meta, compression=compression)  # type: ignore[arg-type]

    assert captured["save_path"].parent == tmp_path
    assert captured["save_path"].suffix == ".tif"
    assert captured["save_path"].name != "out.tif"
    assert out_path.exists()
    assert captured["kwargs"]["predictor"] == expected_predictor
    assert captured["kwargs"]["compression"] == compression
    assert captured["kwargs"]["imagej"] is True


@pytest.mark.parametrize("winerror", [None, 32, 33])
def test_save_tiff_recovers_from_busy_rename(monkeypatch, tmp_path, winerror):
    import errno
    from tifffile import imread

    destination = tmp_path / "out.tif"
    destination.write_bytes(b"previous output")
    replace = Path.replace
    attempts = []
    delays = []
    error = OSError(errno.EBUSY if winerror is None else errno.EACCES, "busy")
    if winerror is not None:
        error.winerror = winerror

    def busy_then_replace(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            assert destination.read_bytes() == b"previous output"
            raise error
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", busy_then_replace)
    monkeypatch.setattr(core.time, "sleep", delays.append)
    arr = np.ones((5, 6), dtype=np.uint16)
    meta = SimpleNamespace(imagej_meta={"axes": "YX"}, resolution=None, extratags=[])
    core.save_tiff(arr, destination, meta, compression=None)

    np.testing.assert_array_equal(imread(destination), arr)
    assert len(attempts) == 3
    assert len(set(attempts)) == 1
    assert delays == [0.25, 0.5]
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("error_number, expected_attempts", [(16, 6), (13, 1), (30, 1)])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_save_tiff_preserves_original_error_and_output(
    monkeypatch, tmp_path, error_number, expected_attempts, cleanup_fails,
):
    destination = tmp_path / "out.tif"
    destination.write_bytes(b"previous output")
    error = OSError(error_number, "rename failed")
    attempts = []
    delays = []

    def fail_replace(source, target):
        attempts.append(source)
        raise error

    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(core.time, "sleep", delays.append)
    if cleanup_fails:
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    arr = np.ones((5, 6), dtype=np.uint16)
    meta = SimpleNamespace(imagej_meta={"axes": "YX"}, resolution=None, extratags=[])
    with pytest.raises(OSError) as caught:
        core.save_tiff(arr, destination, meta, compression=None)

    assert caught.value is error
    assert len(attempts) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert sum(delays) <= 7.75
    assert destination.read_bytes() == b"previous output"
    assert attempts[0].exists() == cleanup_fails
