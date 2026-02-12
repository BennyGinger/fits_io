from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from numpy.typing import NDArray
import numpy as np

from fits_io.readers._types import ArrAxis, PixelSize, StatusFlag, Zproj

DEFAULT_FLAG: StatusFlag = "active"
ALLOWED_FLAGS: set[StatusFlag] = {"active", "skip"}

@dataclass
class ImageReader(ABC):
    """Abstract base class for image readers."""

    img_path: Path
    _channel_labels: list[str] | None = field(default=None, kw_only=True)
    
    def _validate_channel_label_override(self) -> None:
        if self._channel_labels is None:
            return

        counts = self.channel_number
        unique = sorted(set(counts))
        if len(unique) != 1:
            raise ValueError(
                "Cannot use a single global channel_labels override because this file has "
                f"different channel counts per series: {counts}. "
                "Either omit channel_labels or provide per-series labels (not supported yet)."
            )

        n = unique[0]
        if len(self._channel_labels) != n:
            raise ValueError(
                f"Length of provided channel_labels ({len(self._channel_labels)}) does not "
                f"match number of channels in image ({n})."
            )
    
    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if this reader supports the file."""
        ...

    @property
    def channel_labels(self) -> list[str] | None:
        """User override if provided, otherwise subclass may return its own labels."""
        return self._channel_labels if self._channel_labels is not None else self._native_channel_labels()

    @abstractmethod
    def _native_channel_labels(self) -> list[str] | None:
        """Subclasses return labels from file (or None if unavailable)."""
        ...
    
    @property
    @abstractmethod
    def axes(self) -> list[str]:
        """Return the axes string for the image data"""
        ...
    
    @property
    @abstractmethod
    def compression_method(self) -> str | None:
        """Return the compression method used for the image data, or None if uncompressed."""
        ...
    
    @property
    @abstractmethod
    def status(self) -> StatusFlag:
        """Return the status of the image for downstream processing (i.e., 'active' or 'skip')."""
        ...
    
    @property
    @abstractmethod
    def channel_number(self) -> list[int]:
        """Return the number of channels in the image for each series, or 1 if not applicable."""
        ...
    
    @property
    @abstractmethod
    def series_number(self) -> int:
        """Return the number of series in the image, or 1 if not applicable."""
        ...
    
    @abstractmethod
    def axis_index(self, axis: ArrAxis) -> list[int | None]:
        """Return a list of indices of the specified axis, or None if not present, for each series."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> list[PixelSize | None]:
        """Return a list of resolution (um per pixel) for (x,y) axes. If none available, return (1.,1.), for each series."""
        ...
    
    @property
    @abstractmethod
    def interval(self) -> float | None:
        """Return the time interval between frames in seconds, or None if not available."""
        ...
    
    @property
    @abstractmethod
    def custom_metadata(self) -> Mapping[str, Any]:
        """Return custom metadata, if any, associated with the custom pipeline saved under extratags, or empty dict if not available."""
        ...
    
    @abstractmethod
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """Return the image data as a NumPy array. If multiple series, return a list of arrays."""
        ...
    
    @abstractmethod
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """Return the selected channel(s) as a NumPy array or list of arrays. Channel can be specified by index or label."""
        ...

    def apply_z_projection(self, arr: NDArray, z_axis: int | None, method: Zproj | None) -> NDArray:
        """
        Return array after applying z-projection along specified axis, if any and method given. Else return original array.
        Args:
            arr: Input array.
            z_axis: Axis index corresponding to Z dimension.
            method: Z-projection method to apply ('max', 'mean' or None).
        Returns:
            NDArray: Projected array or original array.
        """
        
        
        if z_axis is None or method is None:
            return arr
        
        if method == 'max':
            return np.max(arr, axis=z_axis)
        elif method == 'mean':
            return np.mean(arr, axis=z_axis)
        else:
            raise ValueError(f"Unsupported z-projection method: {method}")