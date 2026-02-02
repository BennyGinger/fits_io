from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from numpy.typing import NDArray
import numpy as np

from fits_io.provenance import FITS_TAG, add_provenance_profile, ExportProfile
from fits_io.image_reader import ImageReader, StatusFlag, StatusProfile
from fits_io._types import ExtraTags, PixelSize, PixelDensity

COLOR_MAP = {
        "red":     (1, 0, 0),
        "green":   (0, 1, 0),
        "blue":    (0, 0, 1),
        "cyan":    (0, 1, 1),
        "magenta": (1, 0, 1),
        "yellow":  (1, 1, 0),
        "gray":    (1, 1, 1),
    }

LABEL_TO_COLOR = {
    "blue": 'blue',
    "bfp": 'blue',
    "dapi": 'blue',
    "cyan": 'cyan',
    "cfp": 'cyan',
    "yellow": 'yellow',
    "cy3": 'yellow',
    "green": 'green',
    "gcamp": 'green',
    "gfp": 'green',
    "egfp": 'green',
    "fitc": 'green',
    "magenta": 'magenta',
    "mch": 'magenta',
    "ired": 'magenta',
    "irfp": 'magenta',
    "red": 'red',
    "pinky": 'red',
    "mkate2": 'red',
    "scarlet": 'red',
    "geco": 'red',
    "mcherry": 'red',
    "tritc": 'red',
    "rfp": 'red',
    'gray': 'gray',
    'grey': 'gray',
}



class StackMeta:
    
    def __init__(self, axes: str, status: StatusFlag, finterval: float | None) -> None:
        self.axes = axes
        self._status: StatusFlag = status
        self.finterval = finterval
    
    @property
    def status(self) -> str:
        status_profile = StatusProfile(status=self._status)
        return status_profile.export
    
    def change_status(self, new_status: StatusFlag) -> None:
        self._status = new_status
    
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any]=  {'axes': self.axes,
                             'Info': self.status}
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


def make_color_lut(color: str) -> NDArray[np.uint8]:
    """Return ImageJ-style LUT: shape (3, 256), uint8."""
    
    try:
        mask = np.array(COLOR_MAP[color.lower()], dtype=np.uint8)[:, None]
    except KeyError:
        raise ValueError(f"Unsupported LUT color: {color}")

    return mask * np.arange(256, dtype=np.uint8)   


@dataclass(slots=True)
class ChannelMeta:
    channel_number: int
    labels: str | Sequence[str] | None = None
    mode: str = field(init=False)
    luts: list[NDArray[np.uint8]] | None = field(init=False)
    
    def __post_init__(self):
        if self.labels is None or self.labels == 'initialize':
            self.mode, self.luts = 'grayscale', None
            self.labels = self.init_labels()
            return
        
        if isinstance(self.labels, str):
            self.labels = [self.labels]
        
        if self.labels is not None and len(self.labels) != self.channel_number:
            raise ValueError(f"Expected {self.channel_number} labels, got {len(self.labels)}")
        
        colors = [LABEL_TO_COLOR.get(lbl.lower(), None) for lbl in self.labels]
        if any(c not in COLOR_MAP for c in colors):
            self.mode, self.luts = 'grayscale', None
        else:
            self.mode, self.luts = 'color', [make_color_lut(c) for c in colors if c is not None]
    
    def init_labels(self) -> list[str]:
        return [f"C{i+1}" for i in range(self.channel_number)]
         
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

def _encode_metadata(payload: Mapping[str, Any]) -> ExtraTags | None:
    """
    Encode metadata dictionary as JSON and prepare for storage in TIFF extra tags.
    """
    if payload:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return [(FITS_TAG, "B", len(raw), raw, True)]
    return None

def _update_metadata(original_meta: Mapping[str, Any], *, update_meta: Mapping[str, Any] | None, export_profile: ExportProfile | None = None) -> dict[str, Any]:
    """
    Update original metadata dictionary with values from update_meta.
    """
    out = dict(original_meta)
    
    if update_meta is None:
        return out
    
    if export_profile is not None:
        out[export_profile.step_name].update(update_meta)
    else:
        out.update(update_meta)
    return out

def build_imagej_metadata(img_reader: ImageReader, *, export_profile: ExportProfile | None = None, channel_labels: str | Sequence[str] | None = None, extra_step_metadata: Mapping[str, Any] | None = None, new_status: StatusFlag | None = None, series_index: int = 0) -> TiffMetadata:
    """
    Build ImageJ-compatible metadata for saving TIFF files.
    
    Args:
        img_reader: An ImageReader instance to read metadata from.
        channel_labels: Optional; either a single string label or a sequence of labels for each channel. If None, default labels will be used.
        extra_step_metadata: Optional mapping of additional metadata to include in the processing step.
        new_status: Optional; if provided, overrides the status in the metadata.
        series_index: Optional; index of the series to use for multi-series images, purely to save appropriate metadata.
    
    Returns:
        TiffMetadata object containing metadata, resolution, and extra tags.
    """
    
    input_channel_labels = img_reader.channel_labels
    if channel_labels is not None:
        input_channel_labels = channel_labels
    
    # extract metadata components
    stack_meta = StackMeta(axes=img_reader.axes[series_index], 
                           status=img_reader.status,
                           finterval=img_reader.interval)
    if new_status is not None:
        stack_meta.change_status(new_status)
    channel_meta = ChannelMeta(channel_number=img_reader.channel_number[series_index], labels=input_channel_labels)
    resolution_meta = ResolutionMeta(img_reader.resolution[series_index])
    
    # build final metadata dict
    metadata_dict = stack_meta.to_dict()
    metadata_dict.update(channel_meta.to_dict())
    metadata_dict.update(resolution_meta.to_dict())
    
    # build custom metadata to be stored in private tag
    extratags: ExtraTags | None = None
    
    existing_metadata = img_reader.custom_metadata
    payload = add_provenance_profile(existing_metadata, export_profile=export_profile)
    payload = _update_metadata(payload, update_meta=extra_step_metadata, export_profile=export_profile)
    extratags = _encode_metadata(payload)
    
    return TiffMetadata(
        imagej_meta=metadata_dict,
        resolution=resolution_meta.resolution,
        extratags=extratags)


                          

if __name__ == '__main__':
    from time import time
    from tifffile import imread, imwrite
    
    
    payload = json.dumps({
                'resolution': (1/0.3223335, 1/0.3223335),
                'Some_other_metadata': {'key1':'value1',
                                        'key2':'value2'}
            })
    
    # Test
    t1 = time()
    img_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    save_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif')
    img = imread(img_path)
    imwrite(save_path,
            img,
            imagej=True,
            metadata={
                    'axes':'TZCYX',
                    'finterval':11,
                    'Labels': ["GFP", "mCherry"],
                    'mode':'color',   
                    'LUTs': [make_color_lut("green"), make_color_lut("red")],
                    "unit": "um"
                },
            resolution=(1/0.3223335, 1/0.3223335),
            predictor=2,
            extratags=[(FITS_TAG, "s", 0, payload, True)],
            compression='zlib',
    )
    
    
    t2 = time()
    print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")



