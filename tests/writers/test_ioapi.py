from __future__ import annotations

import numpy as np
import pytest

from fits_io.metadata.models import FitsIOPayload
from fits_io.writers import apis


# -----------------------------------
# convert_to_fits_tif()
# -----------------------------------

def test_convert_to_fits_tif_returns_output_paths(monkeypatch, dummy_reader) -> None:
    out_path = dummy_reader.img_path.with_name("fits.tif")
    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [out_path])
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: None)

    out = apis.convert_to_fits_tif(dummy_reader)

    assert out == [out_path]


def test_convert_to_fits_tif_writes_one_file_per_series(monkeypatch, dummy_reader) -> None:
    s1 = dummy_reader.img_path.with_name("img_s1") / "fits.tif"
    s2 = dummy_reader.img_path.with_name("img_s2") / "fits.tif"

    r1 = type(dummy_reader)(img_path=dummy_reader.img_path, series_idx=0)
    r2 = type(dummy_reader)(img_path=dummy_reader.img_path, series_idx=1)
    monkeypatch.setattr(dummy_reader, "split_series", lambda: [r1, r2])

    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [s1, s2])

    saved: list[dict[str, object]] = []
    monkeypatch.setattr(
        apis,
        "save_tiff",
        lambda array, path, _meta, compression=None: saved.append({"array": array, "path": path, "compression": compression}),
    )

    out = apis.convert_to_fits_tif(dummy_reader, output_name="fits.tif", compression="zlib")

    assert out == [s1, s2]
    assert [x["path"] for x in saved] == [s1, s2]
    assert [x["compression"] for x in saved] == ["zlib", "zlib"]


def test_convert_to_fits_tif_sets_source_channel_identity_for_subset(monkeypatch, dummy_reader) -> None:
    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [dummy_reader.img_path.with_name("fits.tif")])
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: None)

    built: list[dict[str, object]] = []

    def fake_build_payload(base: FitsIOPayload, **kwargs):
        built.append(kwargs)
        return base.with_fitsio(**{k: v for k, v in kwargs.items() if k in {
            "axes", "channel_labels", "n_channels", "source_channel_indices", "source_channel_count", "z_projection", "compression"
        }})

    monkeypatch.setattr(apis, "build_payload", fake_build_payload)

    apis.convert_to_fits_tif(
        dummy_reader,
        channel_labels=["DAPI", "GFP", "RFP"],
        export_channels=["GFP", "RFP"],
    )

    assert built[0]["source_channel_count"] == 3
    assert built[0]["source_channel_indices"] == [1, 2]


def test_convert_to_fits_tif_passes_custom_metadata(monkeypatch, dummy_reader) -> None:
    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [dummy_reader.img_path.with_name("fits.tif")])
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: None)

    seen: list[dict[str, object]] = []

    def fake_build_payload(base: FitsIOPayload, **kwargs):
        seen.append(kwargs)
        return base.with_custom_metadata(kwargs.get("custom_metadata"))

    monkeypatch.setattr(apis, "build_payload", fake_build_payload)

    apis.convert_to_fits_tif(dummy_reader, custom_metadata={"run_id": 7})

    assert seen[0]["custom_metadata"] == {"run_id": 7}


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
