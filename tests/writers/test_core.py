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

    core.save_tiff(arr, tmp_path / "out.tif", meta, compression=compression)  # type: ignore[arg-type]

    assert captured["save_path"].name == "out.tif"
    assert captured["kwargs"]["predictor"] == expected_predictor
    assert captured["kwargs"]["compression"] == compression
    assert captured["kwargs"]["imagej"] is True
