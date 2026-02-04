from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fits.environment.constant_variables import FITS_FILES

# Custom FITS tag number for storing processing provenance metadata in TIFF files
FITS_TAG = 65000


def get_dist_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def add_provenance_profile(custom_metadata: Mapping[str, Any], *, distribution: str, step_name: str) -> dict[str, Any]:
    """
    Small helper to add a provenance profile to custom metadata while saving the TIFF.
    Args:
        custom_metadata: Existing custom metadata mapping.
        distribution: Name of the distribution or package.
        step_name: Name of the processing step.
        
    Returns:
        Updated custom metadata dictionary including the new step.
        """
    
    out = dict(custom_metadata)
    
    out[step_name] = {
        "dist": distribution,
        "version": get_dist_version(distribution),
        "timestamp": utc_now_iso(),
        }
    return out

def series_has_outputs(save_dir: Path) -> bool:
    """Check if a given series directory contains converted FITS arrays.
    
    Args:
        save_dir: Path to the series directory.
    Returns:
        True if the directory exists and contains at least one file, False otherwise.
    """
    if not save_dir.is_dir():
        return False
    
    return any((save_dir / name).is_file() for name in FITS_FILES)

def image_converted(save_dirs: list[Path]) -> bool:
    """
    Check if all series of an image have been converted and saved.
    Args:
        save_dirs: List of Paths to the series directories of an image.
    Returns:
        True if all series directories exist and contain converted FITS arrays, False otherwise.
    """
    return all(series_has_outputs(d) for d in save_dirs)

def is_processed(custom_metadata: Mapping[str, Any], *, step: str) -> bool:
    if not isinstance(custom_metadata, Mapping):
        return False
    return step in custom_metadata

def get_timestamp(custom_metadata: Mapping[str, Any], *, step: str) -> str | None:
    if not is_processed(custom_metadata, step=step):
        return None
    step_metadata = custom_metadata.get(step)
    if not isinstance(step_metadata, Mapping):
        return None
    timestamp = step_metadata.get("timestamp")
    if not isinstance(timestamp, str):
        return None
    return timestamp