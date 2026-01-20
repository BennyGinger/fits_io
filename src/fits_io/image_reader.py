from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, cast

import nd2
from tifffile import TiffFile, TiffPage, imread
from numpy.typing import NDArray


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
    def resolution(self) -> tuple[float, float]:
        """Return the resolution (um per pixel) for (x,y) axes. If none available, return (1.,1.)."""
        ...
    
    @property
    @abstractmethod
    def interval(self) -> float | None:
        """Return the time interval between frames in seconds, or None if not available."""
        ...
    
    @abstractmethod
    def get_array(self) -> NDArray:
        """Return the image data as a NumPy array."""
        ...

@dataclass
class Nd2Reader(ImageReader):
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() == '.nd2'

    @property
    def axes(self) -> str:
        with nd2.ND2File(self.img_path) as file:
            sizes = file.sizes
        return ''.join(sizes.keys())
    
    @property
    def channel_number(self) -> int:
        with nd2.ND2File(self.img_path) as file:
            sizes = file.sizes
        return sizes.get('C', 1)
    
    @property
    def resolution(self) -> tuple[float, float]:
        default_res = (1., 1.)
        
        with nd2.ND2File(self.img_path) as file:
            meta = file.metadata
        
        channels = getattr(meta, 'channels', None)
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
        with nd2.ND2File(self.img_path) as file:
            exp = file.experiment
            sizes = file.sizes
        
        if sizes.get('T', 1)>1:
            exploop = exp[0]
            if exploop.type == 'TimeLoop': #for normal timelapse experiments
                return int(exploop.parameters.periodMs/1000)
            elif exploop.type == 'NETimeLoop': #for ND2 Merged Experiments
                return int(exploop.parameters.periods[0].periodMs/1000)
        else:
            return None
    
    def get_array(self) -> NDArray:
        return nd2.imread(self.img_path)
    
@dataclass
class TiffReader(ImageReader):
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff']

    @property
    def axes(self) -> str:
        with TiffFile(self.img_path) as tif:
            return tif.series[0].axes
    
    @property
    def channel_number(self) -> int:
        with TiffFile(self.img_path) as tif:
            shape = tif.series[0].shape
            axes = tif.series[0].axes
        if 'C' in axes:
            c_index = axes.index('C')
            return shape[c_index]
        else:
            return 1
    
    @property
    def resolution(self) -> tuple[float, float]:
        x_um_per_pix, y_um_per_pix = (1., 1.)
        with TiffFile(self.img_path) as tif:
            page0 = cast(TiffPage, tif.pages[0]) # for typing purposes
            for tag in page0.tags:
                if tag.name == 'XResolution':
                    xres = tag.value[0]/tag.value[1]
                    x_um_per_pix = round(1./float(xres), 4)
                if tag.name == 'YResolution':
                    yres = tag.value[0]/tag.value[1]
                    y_um_per_pix = round(1./float(yres), 4)
        return (x_um_per_pix, y_um_per_pix)
                
    @property
    def interval(self) -> float | None:
        with TiffFile(self.img_path) as tif:
            imgj_meta = tif.imagej_metadata or {}
        return imgj_meta.get('finterval', None)
    
    def get_array(self) -> NDArray:
        return imread(self.img_path)


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
    # Test writing tiff with extra metadata
    from pathlib import Path
    import numpy as np
    import nd2
    
    tif_path = Path('/home/ben/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    
    rtif = TiffReader(tif_path)
    print(rtif.channel_number)
    # print(rtif.resolution)
    # print(rtif.interval)
    # print(rtif.axes)
    # print(rtif.get_array().shape)
        
    
    
    nd2_path = Path('/home/ben/Docker_mount/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2')
    
    rnd2 = get_reader(nd2_path)
    # rnd2 = Nd2Reader(nd2_path)
    print(rnd2.resolution)
    print(rnd2.interval)
    print(rnd2.axes)
    print(rnd2.get_array().shape)
    