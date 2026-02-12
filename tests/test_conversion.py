# tests/test_conversion.py
from __future__ import annotations

import numpy as np
import pytest

from fits_io.writers.utils import resolve_channel_labels, get_array_to_export


# -----------------------------
# resolve_channel_labels tests
# -----------------------------

def test_resolve_channel_labels_none_defaults_and_exports_all_flag_true():
    labels, export_all = resolve_channel_labels(
        channel_labels=None, n_channels=3, export_channels="all"
    )
    assert labels == ["C_1", "C_2", "C_3"]
    assert export_all is True


def test_resolve_channel_labels_str_single_channel_ok_and_all_exports():
    labels, export_all = resolve_channel_labels(
        channel_labels="RFP", n_channels=1, export_channels="all"
    )
    assert labels == ["RFP"]
    assert export_all is True


def test_resolve_channel_labels_list_subset_success():
    labels, export_all = resolve_channel_labels(
        channel_labels=["GFP", "RFP"],
        n_channels=2,
        export_channels=["RFP"],
    )
    assert labels == ["RFP"]
    assert export_all is False


def test_resolve_channel_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        resolve_channel_labels(
            channel_labels=["GFP"],
            n_channels=2,
            export_channels="all",
        )


def test_resolve_channel_labels_wrong_type_channel_labels_raises():
    with pytest.raises(TypeError):
        resolve_channel_labels(  
            channel_labels=123, # type: ignore[arg-type]
            n_channels=2,
            export_channels="all",
        )


# -----------------------------
# get_array_to_export tests
# -----------------------------

def test_get_array_to_export_all_calls_get_array(dummy_reader):
    dummy_reader.array = np.ones((4, 4), dtype=np.uint8)

    out = get_array_to_export(
        dummy_reader,
        export_channels=["C_1"],   # ignored in this branch
        export_all_flag=True,
        z_projection="max",
    )

    assert dummy_reader.called_get_array == 1
    assert dummy_reader.called_get_channel == 0
    assert isinstance(out, list)
    assert out[0].shape == (4, 4)


def test_get_array_to_export_subset_calls_get_channel(dummy_reader):
    dummy_reader.channel_array = np.ones((5, 5), dtype=np.uint8)

    out = get_array_to_export(
        dummy_reader,
        export_channels=["C_2"],
        export_all_flag=False,
        z_projection=None,
    )

    assert dummy_reader.called_get_array == 0
    assert dummy_reader.called_get_channel == 1
    assert dummy_reader.last_get_channel_arg == ["C_2"]
    assert len(out) == 1
    assert out[0].shape == (5, 5)


def test_get_array_to_export_empty_array_raises(dummy_reader):
    dummy_reader.array = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="Export produced empty arrays"):
        get_array_to_export(
            dummy_reader,
            export_channels=["C_1"],
            export_all_flag=True,
        )
