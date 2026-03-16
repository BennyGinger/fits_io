from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from fits_io.metadata.provenance import add_provenance_profile
from fits_io.readers._types import StatusFlag, Zproj
from fits_io.readers.protocol import ALLOWED_FLAGS, DEFAULT_FLAG

if TYPE_CHECKING:
    from fits_io.metadata.context import MetadataBuildContext


DEFAULT_STEP_NAME = 'unknown_step_1'
DEFAULT_DISTRIBUTION = 'unknown_distribution'


def get_step_name(original_meta: Mapping[str, Any], *, step_name: str | None) -> str:
    """
    Resolve processing step name, auto-incrementing unknown step names.
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

def get_status(original_meta: Mapping[str, Any]) -> StatusFlag:
    """
    Return valid status from metadata, defaulting to active/DEFAULT_FLAG.
    """
    status = original_meta.get('status', DEFAULT_FLAG)
    return status if status in ALLOWED_FLAGS else DEFAULT_FLAG

def build_private_payload(ctx: MetadataBuildContext, *, add_step_meta: bool = True, distribution: str | None = None, extra_step_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None) -> dict[str, Any]:
    """
    Build the final private metadata payload from resolved context and call-time payload options.
    """
    payload = dict(ctx.base_payload)
    if add_step_meta:
        dist = distribution or DEFAULT_DISTRIBUTION
        payload = add_provenance_profile(payload, distribution=dist, step_name=ctx.step_name)
    if ctx.source_channel_indices is not None:
        payload['source_channel_indices'] = ctx.source_channel_indices
        payload['source_channel_count'] = ctx.source_channel_count
    return _update_metadata(payload, update_meta=extra_step_metadata, user_name=ctx.user_name, step_name=ctx.step_name, z_projection=z_projection, status=ctx.status)

def _update_metadata(original_meta: Mapping[str, Any], *, update_meta: Mapping[str, Any] | None, user_name: str, step_name: str, z_projection: Zproj, status: StatusFlag) -> dict[str, Any]:
    """Merge step metadata and enforce top-level private metadata policy keys."""
    out = dict(original_meta)
    meta = dict(update_meta) if update_meta else {}
    out['user_name'] = user_name
    out['status'] = status
    out['z_projection_method'] = z_projection
    if not meta:
        return out
    out[step_name] = _merge_step_metadata(out.get(step_name), meta)
    return out

def _merge_step_metadata(existing_step_meta: Any, update_meta: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing_step_meta) if isinstance(existing_step_meta, Mapping) else {}

    if 'channels' in update_meta:
        new_channels = update_meta.get('channels')
        if isinstance(new_channels, Mapping):
            existing_channels = merged.get('channels')
            merged_channels = dict(existing_channels) if isinstance(existing_channels, Mapping) else {}
            merged_channels.update(dict(new_channels))
            merged['channels'] = merged_channels
        else:
            merged['channels'] = new_channels

    for key, value in update_meta.items():
        if key == 'channels':
            continue
        merged[key] = value
    return merged