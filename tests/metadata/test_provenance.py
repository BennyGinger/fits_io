from datetime import datetime, timezone

from fits_io.metadata.provenance import _get_dist_version, add_provenance_profile, _utc_now


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
    """Should return timestamp string in expected format."""
    timestamp = _utc_now()
    
    assert isinstance(timestamp, str)
    # Should be in format: YYYY-MM-DD HH:MM:SS
    assert len(timestamp) == 19
    # Should be parseable as datetime
    parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    assert parsed is not None


def test_utc_now_iso_is_utc_timezone():
    """Should return timestamp close to current UTC time."""
    timestamp = _utc_now()
    parsed = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    # Check it's close to current UTC time (within 1 second)
    # Note: parsed datetime is naive, so compare with naive UTC time
    now = datetime.now(timezone.utc).replace(tzinfo=None)
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
    """Should include timestamp in expected format."""
    result = add_provenance_profile({}, distribution="my-dist", step_name="step")
    
    timestamp = result["step"]["timestamp"]
    assert isinstance(timestamp, str)
    # Should be parseable in the expected format
    datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

