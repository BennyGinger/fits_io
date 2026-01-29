from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tifffile import TiffFile


def read_tiff_channels(path: str | Path, channel: int | str | Sequence[int | str], *, channel_labels: Sequence[str] | None = None) -> NDArray:
    """
    Read one or more channels from a TIFF hyperstack stored as pages (e.g. TZCYX).

    Rules:
    - single channel  -> C axis is dropped
    - multiple        -> C axis is preserved (length = number of channels)

    Parameters
    ----------
    path:
        TIFF file path.
    channel:
        Channel selector(s): int indices and/or str labels (all must be same type).
    channel_labels:
        Required if channel is/contains str. Used to map label -> index.

    Returns
    -------
    NDArray
        Array with the requested channels.
    """
    # normalize input
    chan_list: list[int | str] = (
        [channel] if isinstance(channel, (int, str)) else list(channel)
    )
    if not chan_list:
        raise ValueError("channel selection cannot be empty")

    # resolve to indices
    if any(isinstance(c, str) for c in chan_list):
        if not all(isinstance(c, str) for c in chan_list):
            raise TypeError("Cannot mix int and str channel selectors.")
        if not channel_labels:
            raise ValueError("channel_labels must be provided when selecting by name.")
        labels = list(channel_labels)
        try:
            c_list = [labels.index(name) for name in chan_list]  # type: ignore[arg-type]
        except ValueError as e:
            raise ValueError(f"Unknown channel in {chan_list!r}. Available: {labels}") from e
    else:
        c_list = [int(c) for c in chan_list]  # type: ignore[arg-type]

    # keep order, drop duplicates
    seen: set[int] = set()
    c_list = [c for c in c_list if not (c in seen or seen.add(c))]

    with TiffFile(path) as tif:
        s = tif.series[0]
        axes = s.axes
        shape = s.shape

        if "C" not in axes:
            raise ValueError(f"No 'C' axis in TIFF axes={axes!r}")

        c_dim = shape[axes.index("C")]
        for c in c_list:
            if not (0 <= c < c_dim):
                raise IndexError(f"channel index {c} out of range (0..{c_dim-1})")

        # page axes define ordering; each page is typically YX
        page_axes = [ax for ax in axes if ax not in ("Y", "X")]
        page_shape = [shape[axes.index(ax)] for ax in page_axes]

        # index vectors for each page axis (C fixed to c_list)
        idx_grids: list[np.ndarray] = []
        for ax, n in zip(page_axes, page_shape):
            if ax == "C":
                idx_grids.append(np.array(c_list, dtype=np.int64))
            else:
                idx_grids.append(np.arange(n, dtype=np.int64))

        # enumerate pages
        mesh = np.meshgrid(*idx_grids, indexing="ij")
        multi_idx = np.stack([m.ravel() for m in mesh], axis=0)
        page_indices = np.ravel_multi_index(multi_idx, dims=page_shape, order="C")

        planes = tif.asarray(key=page_indices.tolist())  # (n_planes, Y, X)

        out_page_shape = [
            len(c_list) if ax == "C" else shape[axes.index(ax)]
            for ax in page_axes
        ]
        y = shape[axes.index("Y")]
        x = shape[axes.index("X")]

        out = planes.reshape(*out_page_shape, y, x)

        # always drop C if single channel
        if len(c_list) == 1:
            out = np.squeeze(out, axis=page_axes.index("C"))

        return out