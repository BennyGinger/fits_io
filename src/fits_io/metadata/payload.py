from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Mapping

from fits_io.readers._types import Zproj

if TYPE_CHECKING:
    from fits_io.metadata.context import MetadataBuildContext


def _get_fits_io_version() -> str:
    try:
        return version("fits_io")
    except PackageNotFoundError:
        return "unknown"


def build_private_payload(ctx: MetadataBuildContext, *, project_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = None) -> dict[str, Any]:
    """
    Build private metadata payload from generic context and caller-owned metadata.
    """
    payload = dict(ctx.base_payload)
    existing_fits_io_raw = payload.get("fits_io")
    existing_fits_io = dict(existing_fits_io_raw) if isinstance(existing_fits_io_raw, Mapping) else {}
    src_chan_indices = (
        ctx.source_channel_indices
        if ctx.source_channel_indices is not None
        else existing_fits_io.get("source_channel_indices")
    )
    src_chan_count = (
        ctx.source_channel_count
        if ctx.source_channel_count is not None
        else existing_fits_io.get("source_channel_count")
    )

    payload["fits_io"] = {
        **existing_fits_io,
        "version": _get_fits_io_version(),
        "axes": ctx.axes,
        "channel_labels": ctx.labels,
        "n_channels": ctx.n_channels,
        "z_projection": z_projection,
        "compression": compression,
        "source_channel_indices": src_chan_indices,
        "source_channel_count": src_chan_count,
    }
    if project_metadata:
        payload["project_metadata"] = dict(project_metadata)
    return payload
