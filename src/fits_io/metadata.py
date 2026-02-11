from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import logging

from numpy.typing import NDArray
import numpy as np

from fits_io.provenance import FITS_TAG, add_provenance_profile
from fits_io.image_reader import ImageReader, StatusFlag, InfoProfile, Zproj
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

DEFAULT_STEP_NAME = 'unknown_step_1'
DEFAULT_DISTRIBUTION = 'unknown_distribution'

logger = logging.getLogger(__name__)

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


def _encode_metadata(payload: Mapping[str, Any]) -> ExtraTags | None:
    """
    Encode metadata dictionary as JSON and prepare for storage in TIFF extra tags.
    """
    if payload:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return [(FITS_TAG, "B", len(raw), raw, True)]
    return None

def _update_metadata(original_meta: Mapping[str, Any], *, update_meta: Mapping[str, Any] | None, step_name: str, z_projection: Zproj) -> dict[str, Any]:
    """
    Update original metadata dictionary with values from update_meta.
    """
    out = dict(original_meta)
    meta = dict(update_meta) if update_meta else {}
    
    if not meta:
        return out
    
    if z_projection is not None:
        meta['z_projection_method'] = z_projection
    
    if step_name in out:
        out[step_name].update(meta)
    else:
        out[step_name] = meta
    return out

def _get_step_name(original_meta: Mapping[str, Any], *, step_name: str | None) -> str:
    """
    Get the appropriate step name for provenance tracking.
    
    Args:
        original_meta: Existing custom metadata mapping.
        step_name: Optional desired step name.
    """
    step = step_name or DEFAULT_STEP_NAME
    
    if step == DEFAULT_STEP_NAME:
        meta_keys = original_meta.keys()
        prefix = DEFAULT_STEP_NAME.rsplit("_", 1)[0]
        unknown_keys = [k for k in meta_keys if k.startswith(prefix)]
        
        numbers = [int(k.split("_")[-1]) for k in unknown_keys if k.split("_")[-1].isdigit()]
        next_instance = max(numbers) + 1 if numbers else 1
        step = f"{prefix}_{next_instance}"
    
    return step

def _normalize_channel_labels(labels: str | Sequence[str] | None, n_channels: int) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        if n_channels != 1:
            raise ValueError(f"Expected {n_channels} channel labels, got a single string.")
        return [labels]
    labels_list = list(labels)
    if len(labels_list) != n_channels:
        raise ValueError(f"Expected {n_channels} channel labels, got {len(labels_list)}.")
    return labels_list

def build_imagej_metadata(img_reader: ImageReader, *, user_name: str, distribution: str | None = None, step_name: str | None = None, channel_labels: str | Sequence[str] | None = None, z_projection: Zproj = None, extra_step_metadata: Mapping[str, Any] | None = None, add_provenance: bool = True, new_status: StatusFlag | None = None, series_index: int = 0) -> TiffMetadata:
    """
    Build ImageJ-compatible metadata for saving TIFF files.
    
    Args:
        img_reader: An ImageReader instance to read metadata from.
        user_name: Name of the user performing the conversion.
        distribution: Optional; name of the distribution or package.
        step_name: Optional; name of the processing step.
        channel_labels: Optional; either a single string label or a sequence of labels for each channel. If None, default labels will be used.
        z_projection: Optional; method of z-projection applied to the image data.
        extra_step_metadata: Optional mapping of additional metadata to include in the processing step.
        add_provenance: Optional; if True, add provenance information to the metadata.
        new_status: Optional; if provided, overrides the status in the metadata.
        series_index: Optional; index of the series to use for multi-series images, purely to save appropriate metadata.
    
    Returns:
        TiffMetadata object containing metadata, resolution, and extra tags.
    """
    
    
    
    # Determine channel labels and number of channels for metadata
    if channel_labels is None:
        # exporting all channels
        chosen_labels = img_reader.channel_labels
        n_channels = img_reader.channel_number[series_index]
    else:
        if isinstance(channel_labels, str):
            n_channels = 1
        else:
            n_channels = len(list(channel_labels))

    chosen_labels = _normalize_channel_labels(channel_labels, n_channels)
    
    # Determine axes string for metadata, adjusting for z-projection and channel export
    axes = img_reader.axes[series_index]
    if z_projection is not None:
        axes = axes.replace('Z', '')  # drop Z axis if z-projection is applied
    if n_channels == 1:
        axes = axes.replace('C', '')  # drop C axis for single channel export
    logger.debug(f"Building metadata with axes: {axes}, channel_labels: {chosen_labels}, z_projection: {z_projection}") 
    
    # extract metadata components
    channel_meta = ChannelMeta(channel_number=n_channels, labels=chosen_labels)
    resolution_meta = ResolutionMeta(img_reader.resolution[series_index])
    stack_meta = StackMeta(axes=axes, 
                           status=img_reader.status,
                           user_name=user_name,
                           finterval=img_reader.interval)
    
    # build custom metadata to be stored in private tag
    payload = dict(img_reader.custom_metadata)
    
    if add_provenance:
        dist = distribution or DEFAULT_DISTRIBUTION
        step = _get_step_name(payload, step_name=step_name)
        payload = add_provenance_profile(payload, distribution=dist, step_name=step)
        payload = _update_metadata(payload, update_meta=extra_step_metadata, step_name=step, z_projection=z_projection)
    
    if new_status is not None:
        stack_meta.change_status(new_status)
    
    extratags = _encode_metadata(payload)
    
    # build final metadata dict
    metadata_dict = stack_meta.to_dict()
    metadata_dict.update(channel_meta.to_dict())
    metadata_dict.update(resolution_meta.to_dict())
    
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



