from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from numpy.typing import NDArray
import numpy as np

from fits_io.provenance import FITS_TAG, add_provenance_profile, ExportProfile
from fits_io.image_reader import ImageReader
from fits_io._types import ExtraTags, PixelSize, PixelDensity


LABEL_TO_COLOR = {
    "green": 'green',
    "gfp": 'green',
    "egfp": 'green',
    "fitc": 'green',
    "red": 'red',
    "mcherry": 'red',
    "tritc": 'red',
    "rfp": 'red',
    "blue": 'blue',
    "bfp": 'blue',
    "dapi": 'blue',
}


@dataclass
class StackMeta:
    axes: str
    finterval: float | None
    
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


def make_color_lut(color: str) -> NDArray[np.uint8]:
    """Return ImageJ-style LUT: shape (3, 256), uint8."""
    lut = np.zeros((3, 256), dtype=np.uint8)
    channel_index = {"red": 0, "green": 1, "blue": 2}[color]
    lut[channel_index] = np.arange(256, dtype=np.uint8)
    return lut   


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
        VALID = {"red","green","blue"}
        if any(c not in VALID for c in colors):
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


def build_imagej_metadata(img_reader: ImageReader, export_profile: ExportProfile, *, channel_labels: str | Sequence[str] | None = None, extra_step_metadata: Mapping[str, Any] | None = None,) -> TiffMetadata:
    """
    Build ImageJ-compatible metadata for saving TIFF files.
    
    Args:
        img_reader: An ImageReader instance to read metadata from.
        channel_labels: Optional; either a single string label or a sequence of labels for each channel. If None, default labels will be used.
        extra_step_metadata: Optional mapping of additional metadata to include in the processing step.
    
    Returns:
        TiffMetadata object containing metadata, resolution, and extra tags.
    """
    
    input_channel_labels = img_reader.channel_labels
    if channel_labels is not None:
        input_channel_labels = channel_labels
    
    # extract metadata components
    stack_meta = StackMeta(axes=img_reader.axes, finterval=img_reader.interval)
    channel_meta = ChannelMeta(channel_number=img_reader.channel_number, labels=input_channel_labels)
    resolution_meta = ResolutionMeta(img_reader.resolution)
    
    # build final metadata dict
    metadata_dict = stack_meta.to_dict()
    metadata_dict.update(channel_meta.to_dict())
    metadata_dict.update(resolution_meta.to_dict())
    
    # build custom metadata to be stored in private tag
    extratags: ExtraTags | None = None
    existing_metadata = img_reader.custom_metadata
    payload = add_provenance_profile(existing_metadata, export_profile=export_profile)
    if extra_step_metadata is not None:
        payload[export_profile.step_name].update(extra_step_metadata)
    if payload:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        extratags = [(FITS_TAG, "B", len(raw), raw, True)]
    
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



