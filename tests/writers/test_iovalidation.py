from __future__ import annotations

import pytest

from fits_io.metadata.resolve import resolve_channel_selection


def test_resolve_channel_selection_none_defaults_all():
    sel = resolve_channel_selection(channel_labels=None, n_channels=3, export_channels="all")
    assert sel.source_labels == ["C_1", "C_2", "C_3"]
    assert sel.export_labels == ["C_1", "C_2", "C_3"]
    assert sel.export_indices == [0, 1, 2]


def test_resolve_channel_selection_subset_success():
    sel = resolve_channel_selection(
        channel_labels=["GFP", "RFP", "BFP"],
        n_channels=3,
        export_channels=["RFP", "BFP"],
    )
    assert sel.export_labels == ["RFP", "BFP"]
    assert sel.export_indices == [1, 2]


def test_resolve_channel_selection_export_missing_label_raises():
    with pytest.raises(ValueError, match="not found"):
        resolve_channel_selection(
            channel_labels=["GFP", "RFP"],
            n_channels=2,
            export_channels=["BFP"],
        )


def test_resolve_channel_selection_duplicate_exports_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_channel_selection(
            channel_labels=["GFP", "RFP"],
            n_channels=2,
            export_channels=["RFP", "RFP"],
        )
