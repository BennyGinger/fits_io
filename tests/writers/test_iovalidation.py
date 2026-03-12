# tests/writers/test_validation.py
from __future__ import annotations

from pathlib import Path

import pytest

from fits_io.writers.validation import resolve_channel_labels


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


def test_resolve_channel_labels_fallback_when_requested_not_in_labels():
    """When requested channels are not in labels, fallback to all channels with warning."""
    labels, export_all = resolve_channel_labels(
        channel_labels=["GFP", "RFP"],
        n_channels=2,
        export_channels=["BFP"],  # not in labels
    )
    assert labels == ["GFP", "RFP"]
    assert export_all is True


def test_resolve_channel_labels_list_all_channels_exports_all_false():
    """When explicitly listing all channels, export_all should be False."""
    labels, export_all = resolve_channel_labels(
        channel_labels=["GFP", "RFP", "BFP"],
        n_channels=3,
        export_channels=["GFP", "RFP", "BFP"],
    )
    assert labels == ["GFP", "RFP", "BFP"]
    assert export_all is False



