from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from fits_io.metadata.models import FitsIOPayload
import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import PixelSize, Zproj

@dataclass
class ImageReader(ABC):
    """Abstract base class for image readers."""

    img_path: Path
    series_idx: int = 0
    _shape: tuple[int, ...] = field(init=False)
    _axes: str = field(init=False)
    
    
    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if this reader supports the file."""
        ...

    @property
    @abstractmethod
    def has_series(self) -> bool:
        """Return True if the image has multiple series."""
        ...
    
    @abstractmethod
    def split_series(self) -> list[ImageReader]:
        """Return a list of ImageReader instances, one for each series."""
        ...
    
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the image data of each series."""
        ...
    
    @property
    @abstractmethod
    def axes(self) -> str:
        """Return the axes string for the image data"""
        ...
    
    @property
    @abstractmethod
    def compression_method(self) -> str | None:
        """Return the compression method used for the image data, or None if uncompressed."""
        ...
    
    @property
    @abstractmethod
    def zproj_method(self) -> Zproj:
        """Return the z-projection method applied to the image data, or None if not applicable."""
        ...
    
    @property
    @abstractmethod
    def channel_count(self) -> int:
        """Return the number of channels in the image, or 1 if not applicable."""
        ...
    
    @property
    @abstractmethod
    def series_count(self) -> int:
        """Return the number of series in the image, or 1 if not applicable."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> PixelSize | None:
        """Return the resolution (um per pixel) for (x,y) axes. If not available, return None."""
        ...
    
    @property
    @abstractmethod
    def interval(self) -> float | None:
        """Return the time interval between frames in seconds, or None if not available."""
        ...
    
    @property
    @abstractmethod
    def metadata(self) -> FitsIOPayload:
        """Return array metadata, including fits_io metadata and custom metadata."""
        ...
    
    @abstractmethod
    def get_array(self, z_projection: Zproj = None) -> NDArray[Any]:
        """Return the image data as a NumPy array. If multiple series are present, it will return the first series' array. """
        ...
    
    def _normalize_channel_indices(self, channel_indices: int | Sequence[int]) -> list[int]:
        """
        Make sure that channel_indices is a list of ints, and remove duplicates while preserving order.
        """
        c_list = [channel_indices] if isinstance(channel_indices, int) else list(channel_indices)
        if not c_list:
            raise ValueError("channel selection cannot be empty")
        
        for c in c_list:
            if not isinstance(c, int):
                raise TypeError(f"channel indices must be int, got {type(c).__name__}")
        
        seen: set[int] = set() # Removes duplicates while preserving order
        return [c for c in c_list if not (c in seen or seen.add(c))]
    
    
    def _validate_channel_indices(self, normalized_channel_indices: Sequence[int]) -> None:
        """
        Validate that the provided channel indices are within the valid range for the image's channel count.
        """
        for c in normalized_channel_indices:
            if not (0 <= c < self.channel_count):
                raise IndexError(f"channel index {c} out of range (0..{self.channel_count - 1})")
    
    @abstractmethod
    def get_channel(self, channel: int | Sequence[int], z_projection: Zproj = None) -> NDArray[Any]:
        """Return the selected channel(s) as a NumPy array or list of arrays. Channel can be specified by index or label."""
        ...

    @staticmethod
    def apply_zproj(arr: NDArray, z_axis: int | None, zproj: Zproj | None) -> NDArray:
        """
        Apply z-projection to an array along the specified axis.
        
        Args:
            arr: Input array.
            z_axis: Axis index for Z dimension.
            zproj: Projection method ('max' or 'mean').
        
        Returns:
            Projected array with Z dimension removed.
        """
        if z_axis is None or zproj is None:
            return arr
        
        if zproj == 'max':
            return np.max(arr, axis=z_axis)
        elif zproj == 'mean':
            return np.mean(arr, axis=z_axis)
        else:
            raise ValueError(f"Unsupported z-projection method: {zproj}")
        

if __name__ == "__main__":
    from pathlib import Path
    from tifffile import TiffFile
    
    file_path = Path("/media/ben/Analysis/Python/Images/tiff/Run2/c2z25t23v1_tif.tif")
    
    with TiffFile(file_path) as reader:
        series_list = reader.series
    
    print(f"Number of series: {len(series_list)}")
    print(f"Series shapes: {[s.shape for s in series_list]}")
    try:
        arr = series_list[0].asarray()
        print(f"Array shape: {arr.shape}")
    except Exception as e:
        print(f"Error reading array: {e}")
    
    with TiffFile(file_path) as reader:
        series = reader.series[0]
        arr = series.asarray()
    
    print(f"Array shape: {arr.shape}")
    