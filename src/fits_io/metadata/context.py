from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from fits_io.metadata.arrays import get_channel_count, resolve_axes, validate_labels
from fits_io.readers._types import Zproj
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
    interval: float | None
    resolution: tuple[float, float] | None
    source_channel_indices: list[int] | None = None
    source_channel_count: int | None = None


def resolve_build_context(img_reader: ImageReader, *, channel_labels: str | Sequence[str] | None = None, z_projection: Zproj = None, series_index: int = 0, axis_order: str | None = None, source_channel_indices: list[int] | None = None, source_channel_count: int | None = None) -> MetadataBuildContext:
    """
    Resolve normalized metadata context without mutating payload or building TIFF metadata.
    """
    labels = channel_labels or img_reader.channel_labels
    n_channels = get_channel_count(channel_labels, img_reader.channel_number[series_index])
    labels = validate_labels(labels, n_channels)
    axes = resolve_axes(axis_order=axis_order, reader_axes=img_reader.axes[series_index], z_projection=z_projection, n_channels=n_channels)
    base_payload = dict(img_reader.custom_metadata)
    return MetadataBuildContext(n_channels=n_channels, 
                                labels=labels, 
                                axes=axes, 
                                base_payload=base_payload, 
                                interval=img_reader.interval, 
                                resolution=img_reader.resolution[series_index],
                                source_channel_indices=source_channel_indices, 
                                source_channel_count=source_channel_count)