import pytest

from fits_io.metadata.channels import get_channel_count, validate_labels


def test_validate_labels_none_returns_none():
    assert validate_labels(None, n_channels=3) is None


def test_validate_labels_string_for_single_channel():
    assert validate_labels('GFP', n_channels=1) == ['GFP']


def test_validate_labels_string_for_multi_channel_raises():
    with pytest.raises(ValueError):
        validate_labels('GFP', n_channels=2)


def test_validate_labels_sequence_matches_channel_count():
    assert validate_labels(['GFP', 'RFP'], n_channels=2) == ['GFP', 'RFP']


def test_get_channel_count_from_reader_when_labels_none():
    assert get_channel_count(None, reader_channel_count=4) == 4


def test_get_channel_count_for_single_label_string():
    assert get_channel_count('GFP', reader_channel_count=4) == 1


def test_get_channel_count_for_label_sequence():
    assert get_channel_count(['GFP', 'RFP'], reader_channel_count=4) == 2