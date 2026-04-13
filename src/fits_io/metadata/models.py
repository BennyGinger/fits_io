from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import json

import numpy as np
from numpy.typing import NDArray

from fits_io.readers._types import ExtraTags, PixelDensity, PixelSize
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
    payload: Mapping[str, Any] | None

    @staticmethod
    def _toml_value(value: Any) -> str:
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None:
            return 'null'
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, tuple):
            return json.dumps(list(value), ensure_ascii=False)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(str(value), ensure_ascii=False)

    @classmethod
    def _append_toml_sections(cls, lines: list[str], section: str, value: Mapping[str, Any]) -> None:
        lines.append(f"[{section}]")
        for key, child in value.items():
            if isinstance(child, Mapping):
                continue
            lines.append(f"{key} = {cls._toml_value(child)}")
        lines.append("")
        for key, child in value.items():
            if isinstance(child, Mapping):
                cls._append_toml_sections(lines, f"{section}.{key}", child)
    
    def render(self) -> str:
        delimiter = "----------------------"
        info = [
        delimiter,
        "FITS METADATA",
        delimiter + "\n",
        ]

        if self.payload is None:
            return "\n".join(info) + "\n" + "\n"
        
        fits_meta = self.payload.get("fits_io", {})
        if isinstance(fits_meta, Mapping):
            info.append(f"fits_io version = {fits_meta.get('version', 'unknown')}")

            axes = fits_meta.get("axes")
            if axes is not None:
                info.append(f"axes = {axes}")

            z_proj = fits_meta.get("z_projection")
            if z_proj is not None:
                info.append(f"z_projection = {z_proj}")

            compression = fits_meta.get("compression")
            if compression is not None:
                info.append(f"compression = {compression}")

            labels = fits_meta.get("channel_labels")
            if labels is not None:
                info.append(f"channel labels = {labels}")

            src_idxs = fits_meta.get("source_channel_indices")
            if src_idxs is not None:
                info.append(f"source channel indices = {src_idxs}")

            src_count = fits_meta.get("source_channel_count")
            if src_count is not None:
                info.append(f"source channel count = {src_count}")

        project_meta = self.payload.get("project_metadata", {})
        if isinstance(project_meta, Mapping) and project_meta:
            info.append("\n--- Project Metadata ---\n")
            for key, value in project_meta.items():
                info.append("---")
                if isinstance(value, Mapping):
                    self._append_toml_sections(info, key, value)
                else:
                    info.append(f"{key} = {self._toml_value(value)}")
                    info.append("")
        
        return "\n".join(info) + "\n" + "\n"
            

@dataclass(slots=True)
class TiffMetadata:
    """Container for ImageJ-compatible TIFF metadata."""
    
    imagej_meta: dict[str, Any] = field(default_factory=dict)
    resolution: PixelDensity | None = None
    extratags: ExtraTags | None = None