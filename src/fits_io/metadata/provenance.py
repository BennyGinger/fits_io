from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from typing import Any, Mapping

# Custom FITS tag number for storing processing provenance metadata in TIFF files
FITS_TAG = 65000


def _get_dist_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"

def _utc_now_iso() -> str:
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
        "version": _get_dist_version(distribution),
        "timestamp": _utc_now_iso(),
        }
    return out

