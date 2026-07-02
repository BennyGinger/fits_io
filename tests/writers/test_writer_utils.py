from __future__ import annotations

import numpy as np

from fits_io.writers import apis


def test_convert_to_fits_tif_uses_get_channel_for_subset(monkeypatch, dummy_reader):
    calls = {"get_array": 0, "get_channel": 0}

    def fake_get_array(z_projection=None):
        calls["get_array"] += 1
        return np.ones((3, 4, 4), dtype=np.uint8)

    def fake_get_channel(channel, z_projection=None):
        calls["get_channel"] += 1
        assert channel == [1]
        return np.ones((4, 4), dtype=np.uint8)

    dummy_reader.get_array = fake_get_array
    dummy_reader.get_channel = fake_get_channel

    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [dummy_reader.img_path.with_name("fits.tif")])
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: None)

    apis.convert_to_fits_tif(
        dummy_reader,
        channel_labels=["C_1", "C_2", "C_3"],
        export_channels=["C_2"],
    )

    assert calls["get_channel"] == 1
    assert calls["get_array"] == 0


def test_convert_to_fits_tif_default_uses_get_channel_with_all_indices(monkeypatch, dummy_reader):
    calls = {"get_channel": 0}

    def fake_get_channel(channel, z_projection=None):
        calls["get_channel"] += 1
        assert channel == [0, 1, 2]
        return np.ones((3, 4, 4), dtype=np.uint8)

    dummy_reader.get_channel = fake_get_channel

    monkeypatch.setattr(apis, "build_output_paths", lambda _series, _name: [dummy_reader.img_path.with_name("fits.tif")])
    monkeypatch.setattr(apis, "save_tiff", lambda *_args, **_kwargs: None)

    apis.convert_to_fits_tif(dummy_reader)

    assert calls["get_channel"] == 1
