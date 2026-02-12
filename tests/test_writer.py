# tests/test_writer.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import fits_io.writers.api as writer_mod


# -----------------------
# Low-level: _save_tiff()
# -----------------------

def test__save_tiff_raises_on_empty_array(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {"n": 0}

    def fake_imwrite(*args: Any, **kwargs: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr(writer_mod, "imwrite", fake_imwrite)

    empty = np.array([], dtype=np.uint8)
    meta = SimpleNamespace(imagej_meta={}, resolution=None, extratags=[])

    with pytest.raises(ValueError, match="Cannot save empty array"):
        writer_mod._save_tiff(empty, tmp_path / "out.tif", meta)  # type: ignore[arg-type]

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
def test__save_tiff_predictor_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compression: str | None,
    expected_predictor: int | None,
) -> None:
    captured: dict[str, Any] = {}

    def fake_imwrite(save_path: Path, img_array: np.ndarray, **kwargs: Any) -> None:
        captured["save_path"] = save_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(writer_mod, "imwrite", fake_imwrite)

    arr = np.ones((5, 6), dtype=np.uint16)
    meta = SimpleNamespace(imagej_meta={"axes": "YX"}, resolution=(1.0, 1.0), extratags=[])

    writer_mod._save_tiff(arr, tmp_path / "out.tif", meta, compression=compression)  # type: ignore[arg-type]

    assert captured["save_path"].name == "out.tif"
    assert captured["kwargs"]["predictor"] == expected_predictor
    assert captured["kwargs"]["compression"] == compression
    assert captured["kwargs"]["imagej"] is True


# -----------------------------------
# High-level: convert_to_fits_tif()
# -----------------------------------

def test_convert_to_fits_tif_skips_if_already_converted(writer_harness, dummy_reader) -> None:
    writer_harness.already_converted = True

    out = writer_mod.convert_to_fits_tif(dummy_reader, overwrite=False)

    # per writer.py: returns save_dirs (not save paths) when skipping
    assert out == writer_harness.save_dirs
    assert writer_harness.saved == []


def test_convert_to_fits_tif_writes_one_file_per_series(writer_harness, dummy_reader) -> None:
    s1 = writer_harness.tmp_path / "img_s1"
    s2 = writer_harness.tmp_path / "img_s2"
    writer_harness.save_dirs = [s1, s2]

    a1 = np.ones((3, 3), dtype=np.uint8)
    a2 = np.ones((3, 3), dtype=np.uint8) * 2
    writer_harness.arrays = [a1, a2]

    out_paths = writer_mod.convert_to_fits_tif(
        dummy_reader,
        output_name="fits.tif",
        compression="zlib",
        overwrite=True,
    )

    assert out_paths == [s1 / "fits.tif", s2 / "fits.tif"]
    assert [c["path"] for c in writer_harness.saved] == [s1 / "fits.tif", s2 / "fits.tif"]
    assert [c["array"] for c in writer_harness.saved] == [a1, a2]
    assert [c["compression"] for c in writer_harness.saved] == ["zlib", "zlib"]

    # sanity: metadata built for both series_index values
    assert [kw.get("series_index") for kw in writer_harness.md_calls] == [0, 1]
