from __future__ import annotations

import pytest

from fits_io.writers import apis
from fits_io.metadata.resolve import resolve_channel_selection


def _set_channel_array(dummy_reader, channel_count: int) -> None:
    """Give the shared dummy reader internally consistent CYX data."""
    import numpy as np

    dummy_reader._axes = "CYX"
    dummy_reader.channel_array = np.ones(
        (channel_count, 2, 3), dtype=np.uint8)


# -----------------------------------
# prepare_conversion()
# -----------------------------------

def test_prepare_conversion_returns_output_path(monkeypatch, dummy_reader) -> None:
    _set_channel_array(dummy_reader, 3)
    out_path = dummy_reader.img_path.with_name("fits.tif")
    monkeypatch.setattr(
        apis, "build_output_path", lambda _reader, save_name: out_path)
    selection = resolve_channel_selection(None, dummy_reader.channel_count)

    result = apis.prepare_conversion(
        dummy_reader, selection=selection, output_name="fits.tif")

    assert result.output_path == out_path


def test_prepare_conversion_reads_selected_channels(monkeypatch, dummy_reader) -> None:
    _set_channel_array(dummy_reader, 2)
    selection = resolve_channel_selection(
        ["DAPI", "GFP", "RFP"], 3, ["GFP", "RFP"])
    monkeypatch.setattr(
        apis,
        "build_output_path",
        lambda _reader, save_name: dummy_reader.img_path.with_name(save_name),)

    apis.prepare_conversion(
        dummy_reader, selection=selection, output_name="fits.tif")

    assert dummy_reader.last_get_channel_arg == [1, 2]


def test_prepare_conversion_sets_channel_lineage(monkeypatch, dummy_reader) -> None:
    _set_channel_array(dummy_reader, 2)
    selection = resolve_channel_selection(
        ["DAPI", "GFP", "RFP"], 3, ["GFP", "RFP"])
    monkeypatch.setattr(
        apis,
        "build_output_path",
        lambda _reader, save_name: dummy_reader.img_path.with_name(save_name),)

    result = apis.prepare_conversion(
        dummy_reader, selection=selection, output_name="fits.tif")

    assert result.metadata.fits_io.source_channel_indices == [0, 1, 2]
    assert result.metadata.fits_io.artifact_channel_indices == [1, 2]


def test_prepare_conversion_passes_custom_metadata(monkeypatch, dummy_reader) -> None:
    _set_channel_array(dummy_reader, 3)
    selection = resolve_channel_selection(None, dummy_reader.channel_count)
    monkeypatch.setattr(
        apis,
        "build_output_path",
        lambda _reader, save_name: dummy_reader.img_path.with_name(save_name),)

    result = apis.prepare_conversion(
        dummy_reader,
        selection=selection,
        output_name="fits.tif",
        custom_metadata={"run_id": 7},)

    assert result.metadata.custom_metadata == {"run_id": 7}


# -----------------------------------
# apply_zproj() / set_channel_labels()
# -----------------------------------

def test_set_channel_labels_raises_on_non_tiff_reader(dummy_reader) -> None:
    with pytest.raises(TypeError, match="only supports .tif/.tiff files"):
        apis.set_channel_labels(dummy_reader, ["DAPI", "GFP"])


def test_apply_zproj_raises_on_non_tiff_reader(dummy_reader) -> None:
    with pytest.raises(TypeError, match="only supports .tif/.tiff files"):
        apis.apply_zproj(dummy_reader, "max")


def test_set_channel_labels_sets_valid_labels(monkeypatch, dummy_reader) -> None:
    monkeypatch.setattr(apis, "TiffReader", type(dummy_reader))

    saved: list[object] = []
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: saved.append(True))

    apis.set_channel_labels(dummy_reader, ["DAPI", "GFP", "RFP"])
    assert len(saved) == 1


def test_apply_zproj_saves_projected_array(monkeypatch, dummy_reader) -> None:
    monkeypatch.setattr(apis, "TiffReader", type(dummy_reader))

    saved: list[object] = []
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: saved.append(True))

    apis.apply_zproj(dummy_reader, "max")
    assert len(saved) == 1
