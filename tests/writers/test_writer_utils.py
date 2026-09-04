from __future__ import annotations

import numpy as np

from fits_io.metadata.resolve import resolve_channel_selection
from fits_io.writers import apis


def test_prepare_conversion_uses_get_channel_for_subset(monkeypatch, dummy_reader):
    calls = {"get_array": 0, "get_channel": 0}
    dummy_reader._axes = "CYX"

    def fake_get_array(z_projection=None):
        calls["get_array"] += 1
        return np.ones((3, 4, 4), dtype=np.uint8)

    def fake_get_channel(channel, z_projection=None):
        calls["get_channel"] += 1
        assert channel == [1]
        return np.ones((4, 4), dtype=np.uint8)

    dummy_reader.get_array = fake_get_array
    dummy_reader.get_channel = fake_get_channel

    monkeypatch.setattr(
        apis,
        "build_output_path",
        lambda _reader, save_name: dummy_reader.img_path.with_name(save_name),)
    selection = resolve_channel_selection(
        ["C_1", "C_2", "C_3"], 3, ["C_2"])

    apis.prepare_conversion(
        dummy_reader,
        selection=selection,
        output_name="fits.tif",)

    assert calls["get_channel"] == 1
    assert calls["get_array"] == 0


def test_prepare_conversion_default_uses_all_channel_indices(monkeypatch, dummy_reader):
    calls = {"get_channel": 0}
    dummy_reader._axes = "CYX"

    def fake_get_channel(channel, z_projection=None):
        calls["get_channel"] += 1
        assert channel == [0, 1, 2]
        return np.ones((3, 4, 4), dtype=np.uint8)

    dummy_reader.get_channel = fake_get_channel

    monkeypatch.setattr(
        apis,
        "build_output_path",
        lambda _reader, save_name: dummy_reader.img_path.with_name(save_name),)
    selection = resolve_channel_selection(None, 3)

    apis.prepare_conversion(
        dummy_reader, selection=selection, output_name="fits.tif")

    assert calls["get_channel"] == 1
