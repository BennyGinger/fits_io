from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import ExtraTags, PixelDensity, PixelSize, StatusFlag
from fits_io.metadata.lut import LABEL_TO_COLOR, COLOR_MAP, make_color_lut


class StackMeta:
    
    def __init__(self, axes: str, finterval: float | None = None) -> None:
        self.axes = axes
        self.finterval = finterval
    
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any]=  {'axes': self.axes}
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



@dataclass(slots=True, frozen=True)
class InfoSummary: 
    status: StatusFlag
    user_name: str
    chosen_labels: list[str] | None
    current_meta: Mapping[str, Any] | None
    
    def render(self) -> str:
        delimiter = "----------------------"
        info = [
        delimiter,
        "FITS PIPELINE METADATA",
        delimiter + "\n",
        f"status: {self.status}",
        f"user: {self.user_name}",
        ]

        labels = "unknown" if self.chosen_labels is None else self.chosen_labels
        info.append(f"channel labels: {labels}")
        
        if self.current_meta is None:
            return "\n".join(info) + "\n" + "\n"
        
        meta = dict(self.current_meta).copy()
        z_proj = meta.pop('z_projection_method', 'None')
        info.append(f"z_projection: {z_proj}")
        
        meta.pop('status', None)
        if not meta:
            return "\n".join(info) + "\n" + "\n"
        
        info.append("\n--- Processed Step ---\n")
        for k, v in meta.items():
            timestamp = v.get('timestamp', "unknown") if isinstance(v, Mapping) else "unknown"
            line = f"{k}: with time stamp of {timestamp}"
            info.append(line)
        
        return "\n".join(info) + "\n" + "\n"
            

@dataclass(slots=True)
class TiffMetadata:
    """Container for ImageJ-compatible TIFF metadata."""
    
    imagej_meta: dict[str, Any] = field(default_factory=dict)
    resolution: PixelDensity | None = None
    extratags: ExtraTags | None = None