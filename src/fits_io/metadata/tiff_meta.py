from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import ExtraTags, PixelDensity, PixelSize
from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.imageJ_meta import InfoSummary, LABEL_TO_COLOR, COLOR_MAP, make_color_lut
from fits_io.metadata.codec import encode_metadata


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


@dataclass(slots=True)
class TiffWriteMeta:
    """Container for ImageJ-compatible TIFF metadata."""
    
    imagej_meta: dict[str, Any] = field(default_factory=dict)
    resolution: PixelDensity | None = None
    extratags: ExtraTags | None = None



def assemble_tiff_metadata(payload: FitsIOMeta, interval: float | None, resolution: PixelSize | None,) -> TiffWriteMeta:
    """
    Assemble ImageJ-compatible TIFF metadata from a FitsIOPayload, interval, and resolution.
    """
    fits_io = payload.fits_io

    channel_meta = ChannelMeta(channel_number=fits_io.channel_count or 1,
                               labels=fits_io.channel_labels)
    resolution_meta = ResolutionMeta(resolution)
    stack_meta = StackMeta(axes=fits_io.axes or "YX",
                           finterval=interval)
    info = InfoSummary(payload=payload.to_info_payload())

    imagej_meta = stack_meta.to_dict()
    imagej_meta["Info"] = info.render(delimiter_levels=3)
    imagej_meta.update(channel_meta.to_dict())
    imagej_meta.update(resolution_meta.to_dict())

    extratags = encode_metadata(payload.dump())

    return TiffWriteMeta(imagej_meta=imagej_meta,
                        resolution=resolution_meta.resolution,
                        extratags=extratags,)




