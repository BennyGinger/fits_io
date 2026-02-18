import pytest
import numpy as np

from fits_io.writers.utils import get_array_to_export


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