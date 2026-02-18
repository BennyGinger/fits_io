import json

import pytest

from fits_io.metadata.utils import (
    encode_metadata,
    update_metadata,
    get_step_name,
    validate_labels,
    DEFAULT_STEP_NAME,
)
from fits_io.metadata.provenance import FITS_TAG


# -------------------------
# encode_metadata
# -------------------------

def test_encode_metadata_with_data():
    """Test encoding metadata produces correct TIFF tag structure."""
    payload = {"key": "value", "number": 42}
    result = encode_metadata(payload)
    
    assert result is not None
    assert len(result) == 1
    
    tag, dtype, count, raw, writeonce = result[0]
    assert tag == FITS_TAG
    assert dtype == "B"
    assert writeonce is True
    
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded == payload


def test_encode_metadata_with_empty_dict():
    """Test that empty dict returns None."""
    result = encode_metadata({})
    assert result is None


def test_encode_metadata_with_nested_data():
    """Test encoding complex nested structures."""
    payload = {
        "step1": {
            "param": "value",
            "nested": {"deep": [1, 2, 3]}
        },
        "step2": {"data": 123}
    }
    result = encode_metadata(payload)
    
    assert result is not None
    raw = result[0][3]
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded == payload
    assert decoded["step1"]["nested"]["deep"] == [1, 2, 3]


def test_encode_metadata_with_unicode():
    """Test encoding Unicode strings."""
    payload = {"name": "Test™", "text": "Ångström"}
    result = encode_metadata(payload)
    
    assert result is not None
    raw = result[0][3]
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded["name"] == "Test™"
    assert decoded["text"] == "Ångström"


def test_encode_metadata_count_matches_raw_length():
    """Test that count field matches actual byte length."""
    payload = {"test": "data with some length"}
    result = encode_metadata(payload)
    
    assert result is not None
    tag, dtype, count, raw, writeonce = result[0]
    assert count == len(raw)


def test_encode_metadata_various_types():
    """Test encoding various Python types."""
    payload = {
        "string": "text",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "dict": {"nested": "value"}
    }
    result = encode_metadata(payload)
    
    assert result is not None
    raw = result[0][3]
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded == payload


# -------------------------
# update_metadata
# -------------------------

def test_update_metadata_adds_new_step():
    """Test adding a new step to empty metadata."""
    original = {}
    update = {"param": "value"}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection=None, status="active")
    
    assert "step1" in result
    assert result["step1"]["param"] == "value"
    assert result["status"] == "active"
    assert result["z_projection_method"] is None


def test_update_metadata_preserves_original():
    """Test that original metadata is not modified."""
    original = {"existing_step": {"data": "old"}}
    update = {"param": "new"}
    
    result = update_metadata(original, update_meta=update, step_name="new_step", z_projection=None, status="active")
    
    # Original should be unchanged
    assert original == {"existing_step": {"data": "old"}}
    # Result should have both
    assert result["existing_step"]["data"] == "old"
    assert result["new_step"]["param"] == "new"
    assert result["status"] == "active"


def test_update_metadata_merges_into_existing_step():
    """Test updating an existing step merges values."""
    original = {"step1": {"param1": "value1"}}
    update = {"param2": "value2"}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection=None, status="active")
    
    assert result["step1"]["param1"] == "value1"
    assert result["step1"]["param2"] == "value2"


def test_update_metadata_overwrites_existing_keys():
    """Test that update overwrites existing keys in the same step."""
    original = {"step1": {"param": "old_value"}}
    update = {"param": "new_value"}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection=None, status="active")
    
    assert result["step1"]["param"] == "new_value"


def test_update_metadata_with_none_update():
    """Test that None update_meta adds status and z_projection to metadata."""
    original = {"step1": {"data": "value"}}
    
    result = update_metadata(original, update_meta=None, step_name="step2", z_projection=None, status="active")
    
    assert result["step1"] == {"data": "value"}
    assert result["status"] == "active"
    assert result["z_projection_method"] is None


def test_update_metadata_with_empty_update():
    """Test that empty dict update adds status and z_projection to metadata."""
    original = {"step1": {"data": "value"}}
    
    result = update_metadata(original, update_meta={}, step_name="step2", z_projection=None, status="skip")
    
    assert result["step1"] == {"data": "value"}
    assert result["status"] == "skip"
    assert result["z_projection_method"] is None


def test_update_metadata_adds_z_projection_when_provided():
    """Test that z_projection is added to metadata at top level."""
    original = {}
    update = {"param": "value"}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection="max", status="active")
    
    assert result["step1"]["param"] == "value"
    assert result["z_projection_method"] == "max"


def test_update_metadata_z_projection_without_extra_meta():
    """Test z_projection is added to metadata even when update_meta is empty."""
    original = {}
    update = {}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection="mean", status="active")
    
    # Empty update adds status and z_projection but not step
    assert result["status"] == "active"
    assert result["z_projection_method"] == "mean"
    assert "step1" not in result


@pytest.mark.parametrize("z_proj", ["max", "min", "mean", "sum"])
def test_update_metadata_various_z_projections(z_proj):
    """Test various z-projection methods are added at top level."""
    original = {}
    update = {"param": "value"}
    
    result = update_metadata(original, update_meta=update, step_name="step1", z_projection=z_proj, status="active")
    
    assert result["z_projection_method"] == z_proj


def test_update_metadata_complex_nested_structures():
    """Test updating with complex nested data."""
    original = {"step1": {"nested": {"deep": "value"}}}
    update = {"new_nested": {"data": [1, 2, 3]}}
    
    result = update_metadata(original, update_meta=update, step_name="step2", z_projection=None, status="active")
    
    assert result["step1"]["nested"]["deep"] == "value"
    assert result["step2"]["new_nested"]["data"] == [1, 2, 3]


def test_update_metadata_multiple_steps():
    """Test metadata with multiple processing steps."""
    original = {}
    
    # Add first step
    result1 = update_metadata(original, update_meta={"p1": "v1"}, step_name="step1", z_projection=None, status="active")
    # Add second step
    result2 = update_metadata(result1, update_meta={"p2": "v2"}, step_name="step2", z_projection="max", status="active")
    # Add to first step again
    result3 = update_metadata(result2, update_meta={"p3": "v3"}, step_name="step1", z_projection="max", status="skip")
    
    assert result3["step1"]["p1"] == "v1"
    assert result3["step1"]["p3"] == "v3"
    assert result3["step2"]["p2"] == "v2"
    assert result3["status"] == "skip"  # Last status wins
    assert result3["z_projection_method"] == "max"


# -------------------------
# get_step_name
# -------------------------

def test_get_step_name_uses_provided_name():
    """Test that provided step_name is returned."""
    result = get_step_name({}, step_name="my_custom_step")
    assert result == "my_custom_step"


def test_get_step_name_default_with_empty_metadata():
    """Test default step name with empty metadata."""
    result = get_step_name({}, step_name=None)
    assert result == "unknown_step_1"


def test_get_step_name_increments_when_default_exists():
    """Test step name is incremented when default already exists."""
    metadata = {"unknown_step_1": {}}
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_2"


def test_get_step_name_finds_max_and_increments():
    """Test finding the maximum unknown step number."""
    metadata = {
        "unknown_step_1": {},
        "unknown_step_3": {},
        "unknown_step_2": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_4"


def test_get_step_name_ignores_non_matching_keys():
    """Test that non-matching keys are ignored."""
    metadata = {
        "custom_step": {},
        "another_step_1": {},
        "unknown_step_1": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_2"


def test_get_step_name_handles_non_numeric_suffixes():
    """Test that non-numeric unknown_step keys are ignored."""
    metadata = {
        "unknown_step_1": {},
        "unknown_step_custom": {},
        "unknown_step_": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_2"


def test_get_step_name_with_gap_in_sequence():
    """Test that gaps in sequence don't affect next number."""
    metadata = {
        "unknown_step_1": {},
        "unknown_step_5": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_6"


def test_get_step_name_preserves_custom_name_even_if_exists():
    """Test custom step name is returned even if it exists."""
    metadata = {"my_step": {"existing": "data"}}
    result = get_step_name(metadata, step_name="my_step")
    assert result == "my_step"


def test_get_step_name_default_constant_matches_logic():
    """Test that DEFAULT_STEP_NAME matches the logic."""
    assert DEFAULT_STEP_NAME == "unknown_step_1"
    
    # Verify the constant works as expected
    metadata = {}
    result = get_step_name(metadata, step_name=None)
    assert result == DEFAULT_STEP_NAME


def test_get_step_name_large_numbers():
    """Test handling of large step numbers."""
    metadata = {
        "unknown_step_99": {},
        "unknown_step_100": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_101"


def test_get_step_name_mixed_prefixes():
    """Test that only unknown_step prefix is counted."""
    metadata = {
        "unknown_step_1": {},
        "known_step_2": {},
        "processing_step_3": {},
    }
    result = get_step_name(metadata, step_name=None)
    assert result == "unknown_step_2"


# -------------------------
# validate_labels
# -------------------------

def test_validate_labels_none_returns_none():
    assert validate_labels(None, n_channels=3) is None


def test_validate_labels_string_with_one_channel():
    assert validate_labels("GFP", n_channels=1) == ["GFP"]


def test_validate_labels_string_with_multiple_channels_raises():
    with pytest.raises(ValueError):
        validate_labels("GFP", n_channels=2)


def test_validate_labels_sequence_matches_channel_count():
    labels = ["GFP", "mCherry"]
    assert validate_labels(labels, n_channels=2) == labels


def test_validate_labels_sequence_wrong_length_raises():
    with pytest.raises(ValueError):
        validate_labels(["GFP"], n_channels=2)