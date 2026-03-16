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

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

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
    existing_step_meta = out.get(step_name)
    step_meta = dict(existing_step_meta) if isinstance(existing_step_meta, Mapping) else {}
    step_meta["dist"] = distribution
    step_meta["version"] = _get_dist_version(distribution)
    step_meta["timestamp"] = _utc_now()
    out[step_name] = step_meta
    return out

