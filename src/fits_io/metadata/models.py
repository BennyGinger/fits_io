from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import ExtraTags, PixelDensity, PixelSize, StatusFlag
from fits_io.readers.info import InfoProfile
from fits_io.metadata.lut import LABEL_TO_COLOR, COLOR_MAP, make_color_lut


class StackMeta:
    
    def __init__(self, axes: str, status: StatusFlag, user_name: str = 'unknown', finterval: float | None = None) -> None:
        self.axes = axes
        self._status: StatusFlag = status
        self.user_name = user_name
        self.finterval = finterval
    
    @property
    def info(self) -> str:
        """Return ImageJ-style Info string for status and user."""
        status_profile = InfoProfile(status=self._status, user=self.user_name)
        return status_profile.export
    
    def change_status(self, new_status: StatusFlag) -> None:
        self._status = new_status
    
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any]=  {'axes': self.axes,
                             'Info': self.info}
        if self.finterval is not None:
            d['finterval'] = self.finterval
        return d


class ResolutionMeta: 
    def __init__(self, resolution: PixelSize | None) -> None:
        self._resolution = resolution # e.g. um/pixel
        self._unit = 'pixel'  # default unit
    
    @property
    def resolution(self) -> PixelDensity | None:
        """Return pixel per unit resulution for imagej (pixel density)"""
        if self._resolution is None:
            return None
        return (1/self._resolution[0], 1/self._resolution[1])
    
    @property
    def pixel_size(self) -> PixelSize | None:
        """Return pixel size in um, if available."""
        return self._resolution
    
    @property
    def unit(self) -> str:
        """Return unit string for ImageJ metadata."""
        if self._resolution is None:
            return 'pixel'
        return 'micron'
    
    def to_dict(self) -> dict[str, Any]:
        return {'unit': self.unit}
    

@dataclass(slots=True)
class ChannelMeta:
    channel_number: int
    labels: str | Sequence[str] | None = None
    mode: str = field(init=False)
    luts: list[NDArray[np.uint8]] | None = field(init=False)
    
    def __post_init__(self):
        if self.labels is None:
            self.mode, self.luts = 'grayscale', None
            return
        
        if isinstance(self.labels, str):
            self.labels = [self.labels]
        
        if len(self.labels) != self.channel_number:
            raise ValueError(f"Expected {self.channel_number} labels, got {len(self.labels)}")
        
        colors = [LABEL_TO_COLOR.get(lbl.lower(), None) for lbl in self.labels]
        if any(c not in COLOR_MAP for c in colors):
            self.mode, self.luts = 'grayscale', None
        else:
            self.mode, self.luts = 'color', [make_color_lut(c) for c in colors if c is not None]
    
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any]=  {}
        d['Labels'] = self.labels
        d['mode'] = self.mode
        if self.luts is not None:
            d['LUTs'] = self.luts
        return d
    
    
@dataclass(slots=True)
class TiffMetadata:
    """Container for ImageJ-compatible TIFF metadata."""
    
    imagej_meta: dict[str, Any] = field(default_factory=dict)
    resolution: PixelDensity | None = None
    extratags: ExtraTags | None = None