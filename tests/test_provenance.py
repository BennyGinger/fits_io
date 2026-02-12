from datetime import datetime, timezone

from fits_io.metadata.provenance import _get_dist_version, get_timestamp, _is_processed, add_provenance_profile, _utc_now


# Tests for get_dist_version
def test_get_dist_version_returns_version_for_installed_package():
    """Should return version string for an installed package."""
    # pytest should be installed in test environment
    version = _get_dist_version("pytest")
    
    assert isinstance(version, str)
    assert version != "unknown"
    assert len(version) > 0


def test_get_dist_version_returns_unknown_for_nonexistent_package():
    """Should return 'unknown' for packages that don't exist."""
    version = _get_dist_version("this-package-definitely-does-not-exist-12345")
    
    assert version == "unknown"


# Tests for utc_now_iso
def test_utc_now_iso_returns_iso_format_string():
    """Should return ISO format timestamp string."""
    timestamp = _utc_now()
    
    assert isinstance(timestamp, str)
    # Should be parseable as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


def test_utc_now_iso_is_utc_timezone():
    """Should return timestamp in UTC timezone."""
    timestamp = _utc_now()
    parsed = datetime.fromisoformat(timestamp)
    
    # Check it's close to current UTC time (within 1 second)
    now = datetime.now(timezone.utc)
    diff = abs((now - parsed).total_seconds())
    assert diff < 1.0


# Tests for add_provenance_profile
def test_add_provenance_profile_adds_step_to_empty_metadata():
    """Should add step metadata to empty dict."""
    result = add_provenance_profile({}, distribution="my-dist", step_name="my_step")
    
    assert "my_step" in result
    assert result["my_step"]["dist"] == "my-dist"
    assert "version" in result["my_step"]
    assert "timestamp" in result["my_step"]


def test_add_provenance_profile_preserves_existing_metadata():
    """Should preserve existing metadata while adding new step."""
    existing = {
        "existing_step": {"some": "data"},
        "other_key": "value",
    }
    result = add_provenance_profile(existing, distribution="my-dist", step_name="new_step")
    
    assert "existing_step" in result
    assert result["existing_step"] == {"some": "data"}
    assert result["other_key"] == "value"
    assert "new_step" in result


def test_add_provenance_profile_does_not_mutate_input():
    """Should not modify the input metadata dict."""
    original = {"key": "value"}
    result = add_provenance_profile(original, distribution="my-dist", step_name="step")
    
    assert "step" not in original
    assert original == {"key": "value"}


def test_add_provenance_profile_includes_timestamp():
    """Should include ISO format timestamp."""
    result = add_provenance_profile({}, distribution="my-dist", step_name="step")
    
    timestamp = result["step"]["timestamp"]
    assert isinstance(timestamp, str)
    # Should be parseable
    datetime.fromisoformat(timestamp)



# Tests for is_processed
def test_is_processed_returns_true_when_step_exists():
    """Should return True when step is in metadata."""
    metadata = {"my_step": {"some": "data"}}
    
    assert _is_processed(metadata, step="my_step") is True


def test_is_processed_returns_false_when_step_missing():
    """Should return False when step is not in metadata."""
    metadata = {"other_step": {"some": "data"}}
    
    assert _is_processed(metadata, step="my_step") is False


def test_is_processed_returns_false_for_empty_metadata():
    """Should return False for empty metadata dict."""
    assert _is_processed({}, step="my_step") is False


def test_is_processed_returns_false_for_non_mapping():
    """Should return False if metadata is not a mapping."""
    assert _is_processed(None, step="my_step") is False # pyright: ignore
    assert _is_processed("not a dict", step="my_step") is False # pyright: ignore
    assert _is_processed([], step="my_step") is False # pyright: ignore


# Tests for get_timestamp
def test_get_timestamp_returns_timestamp_when_exists():
    """Should return timestamp string when step exists with timestamp."""
    metadata = {
        "my_step": {
            "dist": "my-dist",
            "version": "1.0.0",
            "timestamp": "2024-01-15T10:30:00+00:00",
        }
    }
    
    timestamp = get_timestamp(metadata, step="my_step")
    
    assert timestamp == "2024-01-15T10:30:00+00:00"


def test_get_timestamp_returns_none_when_step_missing():
    """Should return None when step doesn't exist."""
    metadata = {"other_step": {"timestamp": "2024-01-15T10:30:00+00:00"}}
    
    timestamp = get_timestamp(metadata, step="my_step")
    
    assert timestamp is None


def test_get_timestamp_returns_none_when_timestamp_missing():
    """Should return None when step exists but has no timestamp."""
    metadata = {"my_step": {"dist": "my-dist", "version": "1.0.0"}}
    
    timestamp = get_timestamp(metadata, step="my_step")
    
    assert timestamp is None


def test_get_timestamp_returns_none_when_step_not_mapping():
    """Should return None when step value is not a mapping."""
    metadata = {"my_step": "not a dict"}
    
    timestamp = get_timestamp(metadata, step="my_step")
    
    assert timestamp is None


def test_get_timestamp_returns_none_when_timestamp_not_string():
    """Should return None when timestamp is not a string."""
    metadata = {"my_step": {"timestamp": 12345}}
    
    timestamp = get_timestamp(metadata, step="my_step")
    
    assert timestamp is None


def test_get_timestamp_returns_none_for_empty_metadata():
    """Should return None for empty metadata."""
    timestamp = get_timestamp({}, step="my_step")
    
    assert timestamp is None