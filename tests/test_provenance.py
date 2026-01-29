from datetime import datetime, timezone

from fits_io.provenance import create_export_profile, DEFAULT_DISTRIBUTION, DEFAULT_STEP_NAME, DEFAULT_FILENAME, get_dist_version, get_timestamp, is_processed, add_provenance_profile, utc_now_iso, ExportProfile

# test create_export_profile function
def test_all_none_uses_defaults():
    """When all parameters are None, should use all defaults."""
    profile = create_export_profile({}, None, None, None)
    
    assert profile.dist_name == DEFAULT_DISTRIBUTION
    assert profile.step_name == DEFAULT_STEP_NAME
    assert profile.filename == DEFAULT_FILENAME


def test_custom_distribution():
    """Custom distribution should override default."""
    profile = create_export_profile({}, "my_dist", None, None)
    
    assert profile.dist_name == "my_dist"
    assert profile.step_name == DEFAULT_STEP_NAME
    assert profile.filename == DEFAULT_FILENAME


def test_custom_step_name():
    """Custom step_name should override default."""
    profile = create_export_profile({}, None, "my_step", None)
    
    assert profile.dist_name == DEFAULT_DISTRIBUTION
    assert profile.step_name == "my_step"
    assert profile.filename == DEFAULT_FILENAME


def test_custom_filename():
    """Custom filename should override default."""
    profile = create_export_profile({}, None, None, "output.tif")
    
    assert profile.dist_name == DEFAULT_DISTRIBUTION
    assert profile.step_name == DEFAULT_STEP_NAME
    assert profile.filename == "output.tif"


def test_all_custom():
    """All custom parameters should be respected."""
    profile = create_export_profile({}, "dist", "step", "file.tif")
    
    assert profile.dist_name == "dist"
    assert profile.step_name == "step"
    assert profile.filename == "file.tif"


def test_increments_unknown_step_when_exists():
    """When unknown_step_N exists in metadata, should increment to N+1."""
    metadata = {
        "fits_io.unknown_step_1": {"some": "data"},
        "fits_io.unknown_step_2": {"more": "data"},
    }
    
    profile = create_export_profile(metadata, None, None, None)
    
    assert profile.step_name == "fits_io.unknown_step_3"


def test_increments_unknown_step_single_existing():
    """When only unknown_step_1 exists, should become unknown_step_2."""
    metadata = {"fits_io.unknown_step_1": {"some": "data"}}
    
    profile = create_export_profile(metadata, None, None, None)
    
    assert profile.step_name == "fits_io.unknown_step_2"


def test_no_increment_if_custom_step_provided():
    """Custom step_name should not trigger incrementing logic."""
    metadata = {"fits_io.unknown_step_1": {"some": "data"}}
    
    profile = create_export_profile(metadata, None, "custom_step", None)
    
    assert profile.step_name == "custom_step"


def test_handles_non_numeric_unknown_steps():
    """Should skip unknown_step keys that don't end with digits."""
    metadata = {
        "fits_io.unknown_step_1": {"some": "data"},
        "fits_io.unknown_step_abc": {"bad": "format"},
    }
    
    profile = create_export_profile(metadata, None, None, None)
    
    assert profile.step_name == "fits_io.unknown_step_2"


def test_handles_empty_metadata():
    """Empty metadata should result in unknown_step_1."""
    profile = create_export_profile({}, None, None, None)
    
    assert profile.step_name == DEFAULT_STEP_NAME


def test_handles_other_metadata_keys():
    """Should ignore metadata keys that don't match unknown_step pattern."""
    metadata = {
        "some_other_key": {"data": "value"},
        "fits_io.convert": {"step": "info"},
        "fits_io.unknown_step_1": {"first": "step"},
    }
    
    profile = create_export_profile(metadata, None, None, None)
    
    assert profile.step_name == "fits_io.unknown_step_2"


def test_handles_non_sequential_unknown_steps():
    """Should increment from max, even if numbers are non-sequential."""
    metadata = {
        "fits_io.unknown_step_1": {},
        "fits_io.unknown_step_5": {},
        "fits_io.unknown_step_3": {},
    }
    
    profile = create_export_profile(metadata, None, None, None)
    
    assert profile.step_name == "fits_io.unknown_step_6"


# Tests for get_dist_version
def test_get_dist_version_returns_version_for_installed_package():
    """Should return version string for an installed package."""
    # pytest should be installed in test environment
    version = get_dist_version("pytest")
    
    assert isinstance(version, str)
    assert version != "unknown"
    assert len(version) > 0


def test_get_dist_version_returns_unknown_for_nonexistent_package():
    """Should return 'unknown' for packages that don't exist."""
    version = get_dist_version("this-package-definitely-does-not-exist-12345")
    
    assert version == "unknown"


# Tests for utc_now_iso
def test_utc_now_iso_returns_iso_format_string():
    """Should return ISO format timestamp string."""
    timestamp = utc_now_iso()
    
    assert isinstance(timestamp, str)
    # Should be parseable as ISO format
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


def test_utc_now_iso_is_utc_timezone():
    """Should return timestamp in UTC timezone."""
    timestamp = utc_now_iso()
    parsed = datetime.fromisoformat(timestamp)
    
    # Check it's close to current UTC time (within 1 second)
    now = datetime.now(timezone.utc)
    diff = abs((now - parsed).total_seconds())
    assert diff < 1.0


# Tests for add_provenance_profile
def test_add_provenance_profile_adds_step_to_empty_metadata():
    """Should add step metadata to empty dict."""
    profile = ExportProfile(dist_name="my-dist", step_name="my_step", filename="test.tif")
    
    result = add_provenance_profile({}, export_profile=profile)
    
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
    profile = ExportProfile(dist_name="my-dist", step_name="new_step", filename="test.tif")
    
    result = add_provenance_profile(existing, export_profile=profile)
    
    assert "existing_step" in result
    assert result["existing_step"] == {"some": "data"}
    assert result["other_key"] == "value"
    assert "new_step" in result


def test_add_provenance_profile_does_not_mutate_input():
    """Should not modify the input metadata dict."""
    original = {"key": "value"}
    profile = ExportProfile(dist_name="my-dist", step_name="step", filename="test.tif")
    
    result = add_provenance_profile(original, export_profile=profile)
    
    assert "step" not in original
    assert original == {"key": "value"}


def test_add_provenance_profile_includes_timestamp():
    """Should include ISO format timestamp."""
    profile = ExportProfile(dist_name="my-dist", step_name="step", filename="test.tif")
    
    result = add_provenance_profile({}, export_profile=profile)
    
    timestamp = result["step"]["timestamp"]
    assert isinstance(timestamp, str)
    # Should be parseable
    datetime.fromisoformat(timestamp)


# Tests for is_processed
def test_is_processed_returns_true_when_step_exists():
    """Should return True when step is in metadata."""
    metadata = {"my_step": {"some": "data"}}
    
    assert is_processed(metadata, step="my_step") is True


def test_is_processed_returns_false_when_step_missing():
    """Should return False when step is not in metadata."""
    metadata = {"other_step": {"some": "data"}}
    
    assert is_processed(metadata, step="my_step") is False


def test_is_processed_returns_false_for_empty_metadata():
    """Should return False for empty metadata dict."""
    assert is_processed({}, step="my_step") is False


def test_is_processed_returns_false_for_non_mapping():
    """Should return False if metadata is not a mapping."""
    assert is_processed(None, step="my_step") is False # pyright: ignore
    assert is_processed("not a dict", step="my_step") is False # pyright: ignore
    assert is_processed([], step="my_step") is False # pyright: ignore


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