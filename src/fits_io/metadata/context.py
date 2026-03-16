from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from fits_io.metadata.axes import resolve_axes
from fits_io.metadata.channels import get_channel_count, validate_labels
from fits_io.metadata.private import get_status, get_step_name
from fits_io.readers._types import StatusFlag, Zproj
from fits_io.readers.protocol import ImageReader


@dataclass(slots=True, frozen=True)
class MetadataBuildContext:
    """
    Normalized metadata inputs resolved from reader state and call-time overrides.
    """
    n_channels: int
    labels: list[str] | None
    axes: str
    base_payload: dict[str, Any]
    step_name: str
    status: StatusFlag
    user_name: str
    interval: float | None
    resolution: tuple[float, float] | None
    source_channel_indices: list[int] | None = None
    source_channel_count: int | None = None


def resolve_build_context(img_reader: ImageReader, *, step_name: str | None = None, channel_labels: str | Sequence[str] | None = None, z_projection: Zproj = None, new_status: StatusFlag | None = None, new_user: str | None = None, series_index: int = 0, axis_order: str | None = None, source_channel_indices: list[int] | None = None, source_channel_count: int | None = None) -> MetadataBuildContext:
    """
    Resolve normalized metadata context without mutating payload or building TIFF metadata.
    """
    labels = channel_labels or img_reader.channel_labels
    n_channels = get_channel_count(channel_labels, img_reader.channel_number[series_index])
    labels = validate_labels(labels, n_channels)
    axes = resolve_axes(axis_order=axis_order, reader_axes=img_reader.axes[series_index], z_projection=z_projection, n_channels=n_channels)
    base_payload = dict(img_reader.custom_metadata)
    resolved_step_name = get_step_name(base_payload, step_name=step_name)
    status = new_status or get_status(base_payload)
    user_name = new_user or base_payload.get('user_name', 'unknown')
    return MetadataBuildContext(n_channels=n_channels, 
                                labels=labels, 
                                axes=axes, 
                                base_payload=base_payload, 
                                step_name=resolved_step_name, 
                                status=status, 
                                user_name=user_name, 
                                interval=img_reader.interval, 
                                resolution=img_reader.resolution[series_index],
                                source_channel_indices=source_channel_indices, 
                                source_channel_count=source_channel_count)