import json
from typing import Mapping, Sequence, Any

from fits_io.readers._types import ExtraTags, Zproj
from fits_io.metadata.provenance import FITS_TAG


DEFAULT_STEP_NAME = 'unknown_step_1'

def encode_metadata(payload: Mapping[str, Any]) -> ExtraTags | None:
    """
    Encode metadata dictionary as JSON and prepare for storage in TIFF extra tags.
    Args:
        payload: A dictionary of metadata to encode.
    Returns:
        A list of extra tags to be stored in the TIFF file, or None if no metadata is provided.
    """
    if payload:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return [(FITS_TAG, "B", len(raw), raw, True)]
    return None

def update_metadata(original_meta: Mapping[str, Any], *, update_meta: Mapping[str, Any] | None, step_name: str, z_projection: Zproj) -> dict[str, Any]:
    """
    Update original metadata dictionary with values from update_meta.
    Args:
        original_meta: The original metadata dictionary to update. 
        update_meta: A dictionary of metadata to merge into the original. 
        step_name: The name of the processing step, used as a key for organizing metadata. 
        z_projection: The method of z-projection applied, which may be added to the metadata. Returns: A new dictionary containing the merged metadata.
    Returns:
        A new dictionary containing the merged metadata.
    """
    out = dict(original_meta)
    meta = dict(update_meta) if update_meta else {}
    
    if not meta:
        return out
    
    if z_projection is not None:
        meta['z_projection_method'] = z_projection
    
    if step_name in out:
        out[step_name].update(meta)
    else:
        out[step_name] = meta
    return out

def get_step_name(original_meta: Mapping[str, Any], *, step_name: str | None) -> str:
    """
    Get the appropriate step name for provenance tracking.
    
    Args:
        original_meta: Existing custom metadata mapping.
        step_name: Optional desired step name.
    
    Returns:
        A step name string that is either the provided step_name or a generated unique name based on existing metadata.
    """
    step = step_name or DEFAULT_STEP_NAME
    
    if step == DEFAULT_STEP_NAME:
        meta_keys = original_meta.keys()
        prefix = DEFAULT_STEP_NAME.rsplit("_", 1)[0]
        unknown_keys = [k for k in meta_keys if k.startswith(prefix)]
        
        numbers = [int(k.split("_")[-1]) for k in unknown_keys if k.split("_")[-1].isdigit()]
        next_instance = max(numbers) + 1 if numbers else 1
        step = f"{prefix}_{next_instance}"
    
    return step

def validate_labels(labels: str | Sequence[str] | None, n_channels: int) -> list[str] | None:
    """Validate and normalize channel labels for metadata.
    
    Args: 
        labels: A single string label, a sequence of string labels, or None. 
        n_channels: The expected number of channels for the image data. 
    
    Returns: 
        A list of string labels if valid, or None if input is None. 
    
    Raises: 
        ValueError: If the number of provided labels does not match n_channels, or if a single string is provided when multiple channels are expected. 
    """
    if labels is None:
        return None
    if isinstance(labels, str):
        if n_channels != 1:
            raise ValueError(f"Expected {n_channels} channel labels, got a single string.")
        return [labels]
    labels_list = list(labels)
    if len(labels_list) != n_channels:
        raise ValueError(f"Expected {n_channels} channel labels, got {len(labels_list)}.")
    return labels_list