import pytest

from fits_io.metadata.resolve import ChannelSelection, resolve_output_axes, resolve_channel_selection


def test_resolve_axes_preserves_reader_axes_when_no_transforms():
    out = resolve_output_axes(reader_axes="TZCYX", z_projection=None, n_channels=2)
    assert out == "TZCYX"


def test_resolve_axes_drops_z_when_projected():
    out = resolve_output_axes(reader_axes="TZCYX", z_projection="max", n_channels=2)
    assert out == "TCYX"


def test_resolve_axes_drops_c_for_single_channel():
    out = resolve_output_axes(reader_axes="TCYX", z_projection=None, n_channels=1)
    assert out == "TYX"
    

def test_resolve_channel_selection_defaults_labels_and_exports_all():
    sel = resolve_channel_selection(None, n_channels=3, export_channels="all")
    assert sel.source_labels == ["C_1", "C_2", "C_3"]
    assert sel.export_labels == ["C_1", "C_2", "C_3"]
    assert sel.export_indices == [0, 1, 2]


def test_resolve_channel_selection_string_label_single_channel():
    sel = resolve_channel_selection("GFP", n_channels=1, export_channels="all")
    assert sel.source_labels == ["GFP"]
    assert sel.export_labels == ["GFP"]
    assert sel.export_indices == [0]


def test_resolve_channel_selection_duplicate_source_labels_raises():
    with pytest.raises(ValueError):
        resolve_channel_selection(["GFP", "GFP"], n_channels=2)


def test_resolve_channel_selection_length_mismatch_falls_back_to_defaults():
    sel = resolve_channel_selection(["GFP"], n_channels=2)
    assert sel.source_labels == ["C_1", "C_2"]
    assert sel.export_labels == ["C_1", "C_2"]
    assert sel.export_indices == [0, 1]


def test_resolve_channel_selection_subset_by_labels():
    sel = resolve_channel_selection(["DAPI", "GFP", "RFP"], n_channels=3, export_channels=["GFP", "RFP"])
    assert sel.source_labels == ["DAPI", "GFP", "RFP"]
    assert sel.export_labels == ["GFP", "RFP"]
    assert sel.export_indices == [1, 2]


def test_resolve_channel_selection_unknown_export_label_raises():
    with pytest.raises(ValueError):
        resolve_channel_selection(["DAPI", "GFP"], n_channels=2, export_channels=["RFP"])


def test_resolve_channel_selection_duplicate_exports_raises():
    with pytest.raises(ValueError):
        resolve_channel_selection(["DAPI", "GFP"], n_channels=2, export_channels=["GFP", "GFP"])


def test_channel_selection_validate_array_with_channel_axis_subset_ok():
    sel = ChannelSelection(
        source_labels=["DAPI", "GFP", "RFP"],
        export_labels=["GFP", "RFP"],
        export_indices=[1, 2],
    )
    sel.validate_array(array_shape=(4, 2, 128, 128), axes="TCYX")


def test_channel_selection_validate_array_without_channel_axis_single_channel_ok():
    sel = ChannelSelection(
        source_labels=["GFP"],
        export_labels=["GFP"],
        export_indices=[0],
    )
    sel.validate_array(array_shape=(5, 128, 128), axes="TYX")


def test_channel_selection_validate_array_raises_on_shape_axes_mismatch():
    sel = ChannelSelection(
        source_labels=["GFP"],
        export_labels=["GFP"],
        export_indices=[0],
    )
    with pytest.raises(ValueError, match="does not match axes"):
        sel.validate_array(array_shape=(128, 128), axes="TYX")


def test_channel_selection_validate_array_raises_without_c_axis_and_nonzero_index():
    sel = ChannelSelection(
        source_labels=["DAPI", "GFP"],
        export_labels=["GFP"],
        export_indices=[1],
    )
    with pytest.raises(ValueError, match="no channel axis"):
        sel.validate_array(array_shape=(64, 64), axes="YX")