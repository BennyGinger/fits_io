from fits_io.readers._types import Zproj


def resolve_axes(*, axis_order: str | None, reader_axes: str, z_projection: Zproj, n_channels: int) -> str:
    """Resolve output axes string for ImageJ metadata."""
    axes = axis_order if axis_order is not None else reader_axes
    if z_projection is not None:
        axes = axes.replace('Z', '')
    if n_channels == 1:
        axes = axes.replace('C', '')
    return axes