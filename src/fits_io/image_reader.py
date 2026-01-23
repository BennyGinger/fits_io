from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Type, cast

import nd2
from nd2.structures import Metadata, ExpLoop
from tifffile import TiffFile, TiffPage, imread, TiffPageSeries
from numpy.typing import NDArray
import numpy as np


@dataclass
class ImageReader(ABC):
    """Abstract base class for image readers."""

    img_path: Path
    
    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Return True if this reader supports the file."""
        ...

    @property
    @abstractmethod
    def axes(self) -> str:
        """Return the axes string for the image data."""
        ...
    
    @property
    @abstractmethod
    def channel_number(self) -> int:
        """Return the number of channels in the image."""
        ...
    
    @property
    @abstractmethod
    def serie_axis_index(self) -> int | None:
        """Return the index of the 'serie' axis in the axes string, or None if not present."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> tuple[float, float]:
        """Return the resolution (um per pixel) for (x,y) axes. If none available, return (1.,1.)."""
        ...
    
    @property
    @abstractmethod
    def interval(self) -> float | None:
        """Return the time interval between frames in seconds, or None if not available."""
        ...
    
    @abstractmethod
    def get_array(self) -> NDArray | list[NDArray]:
        """Return the image data as a NumPy array."""
        ...

@dataclass(slots=True)
class Nd2Reader(ImageReader):
    
    _sizes: Mapping[str, int] = field(init=False)
    _meta: Metadata = field(init=False)
    _exploop: list[ExpLoop] = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() == '.nd2'

    def __post_init__(self) -> None:
        with nd2.ND2File(self.img_path) as file:
            self._sizes = file.sizes
            self._meta = file.metadata
            self._exploop = file.experiment
    
    @property
    def axes(self) -> str:
        return ''.join(self._sizes.keys())
    
    @property
    def channel_number(self) -> int:
        return self._sizes.get('C', 1)
    
    @property
    def serie_axis_index(self) -> int | None:
        if 'P' not in self.axes:
            return None
        return self.axes.index('P')
            
    @property
    def resolution(self) -> tuple[float, float]:
        default_res = (1., 1.)
        
        channels = getattr(self._meta, 'channels', None)
        if not channels:
            return default_res
        
        ch0 = channels[0]
        vol = getattr(ch0, "volume", None)
        if vol is None:
            return default_res

        calib = getattr(vol, "axesCalibration", None)
        if not calib:
            return default_res

        # be defensive about length/type
        x_um_per_pix, y_um_per_pix = calib[:2]
        return (round(float(x_um_per_pix), 4), round(float(y_um_per_pix), 4))
        
    @property
    def interval(self) -> int | None:
        if self._sizes.get('T', 1)>1:
            loop0 = self._exploop[0]
            if loop0.type == 'TimeLoop': #for normal timelapse experiments
                return int(loop0.parameters.periodMs/1000)
            elif loop0.type == 'NETimeLoop': #for ND2 Merged Experiments
                return int(loop0.parameters.periods[0].periodMs/1000)
        else:
            return None
    
    def get_array(self) -> NDArray | list[NDArray]:
        arr = nd2.imread(self.img_path)
        if self.serie_axis_index is None:
            return arr
        
        series_lst = np.split(arr, arr.shape[self.serie_axis_index], axis=self.serie_axis_index)
        return [s.squeeze(axis=self.serie_axis_index) for s in series_lst]
        
    
@dataclass
class TiffReader(ImageReader):
    
    _tifserie0: TiffPageSeries = field(init=False)
    _page0: TiffPage = field(init=False)
    _imageJ_meta: dict[str, Any] = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff']

    def __post_init__(self) -> None:
        with TiffFile(self.img_path) as tif:
            self._tifserie0 = tif.series[0]
            self._page0 = cast(TiffPage, tif.pages[0]) # for typing purposes
            self._imageJ_meta = tif.imagej_metadata or {}
            
    @property
    def axes(self) -> str:
        return self._tifserie0.axes
    
    @property
    def channel_number(self) -> int:
        shape = self._tifserie0.shape
        if 'C' in self.axes:
            c_index = self.axes.index('C')
            return shape[c_index]
        else:
            return 1
    
    @property
    def serie_axis_index(self) -> int | None:
        if 'S' not in self.axes:
            return None
        return self.axes.index('S')
    
    @property
    def resolution(self) -> tuple[float, float]:
        x_um_per_pix, y_um_per_pix = (1., 1.)
        for tag in self._page0.tags:
            if tag.name == 'XResolution':
                xres = tag.value[0]/tag.value[1]
                x_um_per_pix = round(1./float(xres), 4)
            if tag.name == 'YResolution':
                yres = tag.value[0]/tag.value[1]
                y_um_per_pix = round(1./float(yres), 4)
        return (x_um_per_pix, y_um_per_pix)
                
    @property
    def interval(self) -> float | None:
        return self._imageJ_meta.get('finterval', None)
    
    def get_array(self) -> NDArray | list[NDArray]:
        arr = imread(self.img_path)
        if self.serie_axis_index is None:
            return arr
        series_lst = np.split(arr, arr.shape[self.serie_axis_index], axis=self.serie_axis_index)
        return [s.squeeze(axis=self.serie_axis_index) for s in series_lst]


class ImageReaderError(Exception):
    """Base exception for reader-related errors."""


class FileNotFoundError(ImageReaderError):
    pass


class UnsupportedFileTypeError(ImageReaderError):
    pass


READER_BY_SUFFIX: dict[str, Type[ImageReader]] = {
    ".tif": TiffReader,
    ".tiff": TiffReader,
    ".nd2": Nd2Reader,
}

def get_reader(path: str | Path) -> ImageReader:
    """Return an ImageReader instance for the given path, based on suffix."""
    p = Path(path)
    
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Path not found or is not a file: {p}")
    
    suffix = p.suffix.lower()

    try:
        reader_cls = READER_BY_SUFFIX[suffix]
    except KeyError as e:
        supported = ", ".join(sorted(READER_BY_SUFFIX))
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {suffix!r}. Supported: {supported}"
        ) from e

    return reader_cls(p)



if __name__ == "__main__":
    
    tif_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    
    rtif = get_reader(tif_path)
    print(rtif.channel_number)
    print(rtif.serie_axis_index)
    print(rtif.resolution)
    print(rtif.interval)
    print(rtif.axes)
    arrs = rtif.get_array()
    if isinstance(arrs, list):
        print(len(arrs))
        print(arrs[0].shape)
    else:
        print(arrs.shape)
        
    
    
    nd2_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/Anna/241217-rGEM-LTB4/Lib53/Lib53_1dpseed_100nMLTB4_001.nd2')
    
    
    rnd2 = get_reader(nd2_path)
    # # rnd2 = Nd2Reader(nd2_path)
    print(rnd2.resolution)
    print(rnd2.interval)
    print(rnd2.axes)
    print(rnd2.serie_axis_index)
    print(len(rnd2.get_array()))
    