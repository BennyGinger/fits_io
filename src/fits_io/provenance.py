from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from typing import Any, Mapping

# Custom FITS tag number for storing processing provenance metadata in TIFF files
FITS_TAG = 65000
DEFAULT_FILENAME = 'fits.tif'
DEFAULT_STEP_NAME = 'fits_io.unknown_step_1'
DEFAULT_DISTRIBUTION = 'unknown_distribution'

@dataclass(frozen=True)
class ExportProfile:
    """Class representing an export profile for fits image conversion or saving.
    Attributes:
        dist_name: Name of the distribution or package.
        step_name: Name of the processing step.
        filename: Default filename for the exported file.
    """
    dist_name: str 
    step_name: str
    filename: str

def create_export_profile(fits_metadata: Mapping[str, Any], distribution: str | None, step_name: str | None, filename: str | None) -> ExportProfile:
    dist = distribution or DEFAULT_DISTRIBUTION
    file = filename or DEFAULT_FILENAME
    step = step_name or DEFAULT_STEP_NAME
    if step == DEFAULT_STEP_NAME:
        meta_keys = fits_metadata.keys()
        unknown_keys = [k for k in meta_keys if k.startswith("fits_io.unknown_step_")]
        
        if unknown_keys:
            numbers = [int(k.split("_")[-1]) for k in unknown_keys if k.split("_")[-1].isdigit()]
            next_instance = max(numbers) + 1
            step = f"fits_io.unknown_step_{next_instance}"
    
    return ExportProfile(dist_name=dist, step_name=step, filename=file)

def get_dist_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def add_provenance_profile(custom_metadata: Mapping[str, Any], *, export_profile: ExportProfile | None = None) -> dict[str, Any]:
    """
    Small helper to add a provenance profile to custom metadata while saving the TIFF.
    Args:
        custom_metadata: Existing custom metadata mapping.
        export_profile: ExportProfile instance containing distribution and step information.
        
    Returns:
        Updated custom metadata dictionary including the new step.
        """
    
    out = dict(custom_metadata)
    
    if export_profile is not None:
        out[export_profile.step_name] = {
            "dist": export_profile.dist_name,
            "version": get_dist_version(export_profile.dist_name),
            "timestamp": utc_now_iso(),
        }
    return out


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