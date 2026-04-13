from typing import Sequence

from fits_io.readers._types import Zproj


def validate_labels(labels: str | Sequence[str] | None, n_channels: int) -> list[str] | None:
    """
    Validate and normalize channel labels for metadata.
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


def get_channel_count(channel_labels: str | Sequence[str] | None, reader_channel_count: int) -> int:
    """
    Resolve channel count for metadata, mirroring current export policy.
    """
    if channel_labels is None:
        return reader_channel_count
    if isinstance(channel_labels, str):
        return 1
    return len(list(channel_labels))


def resolve_axes(*, axis_order: str | None, reader_axes: str, z_projection: Zproj, n_channels: int) -> str:
    """Resolve output axes string for ImageJ metadata."""
    axes = axis_order if axis_order is not None else reader_axes
    if z_projection is not None:
        axes = axes.replace('Z', '')
    if n_channels == 1:
        axes = axes.replace('C', '')
    return axes