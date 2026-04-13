from __future__ import annotations

from typing import Any, Mapping, Sequence

from fits_io.metadata.context import resolve_build_context
from fits_io.metadata.imagej import build_tiff_metadata
from fits_io.metadata.models import TiffMetadata
from fits_io.metadata.payload import build_private_payload
from fits_io.readers._types import Zproj
from fits_io.readers.protocol import ImageReader


def meta_orchestration(img_reader: ImageReader, *, channel_labels: str | Sequence[str] | None = None, z_projection: Zproj = None, series_index: int = 0, axis_order: str | None = None, source_channel_indices: list[int] | None = None, source_channel_count: int | None = None, project_metadata: Mapping[str, Any] | None = None, compression: str | None = None) -> TiffMetadata:
    """
    Build TIFF metadata using the existing resolve -> payload -> render flow.
    Returns a TiffMetadata object containing the final ImageJ metadata dict and private payload dict.
    """
    ctx = resolve_build_context(
        img_reader,
        channel_labels=channel_labels,
        z_projection=z_projection,
        series_index=series_index,
        axis_order=axis_order,
        source_channel_indices=source_channel_indices,
        source_channel_count=source_channel_count,
    )
    payload = build_private_payload(
        ctx,
        project_metadata=project_metadata,
        z_projection=z_projection,
        compression=compression,
    )
    return build_tiff_metadata(ctx, payload)
