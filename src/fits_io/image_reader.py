from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Type, cast
import logging

from fits_io.tiff_channel_io import read_tiff_channels
import nd2
from nd2.structures import Channel, ExpLoop, ChannelMeta, Volume
from tifffile import TiffFile, TiffPage, imread, TiffPageSeries
from numpy.typing import NDArray
import numpy as np

from fits_io.provenance import FITS_TAG
from fits_io._types import PixelSize


StatusFlag = Literal["active", "skip"]
ALLOWED_FLAGS: set[StatusFlag] = {"active", "skip"}
DEFAULT_FLAG: StatusFlag = "active"

logger = logging.getLogger(__name__)

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
    def status(self) -> str:
        """Return the status of the image for downstream processing (i.e., 'active' or 'skip')."""
        ...
    
    @property
    @abstractmethod
    def export_status(self,) -> str:
        """Return the export status string for the image."""
        ...
    
    @property
    @abstractmethod
    def channel_number(self) -> int:
        """Return the number of channels in the image."""
        ...
    
    @property
    @abstractmethod
    def channel_labels(self) -> list[str] | None:
        """Return the list of channel labels, or None if not available."""
        ...
    
    @property
    @abstractmethod
    def series_number(self) -> int:
        """Return the number of series in the image, or 1 if not applicable."""
        ...
    
    @property
    @abstractmethod
    def serie_axis_index(self) -> int | None:
        """Return the index of the 'serie' axis in the axes string, or None if not present."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> PixelSize | None:
        """Return the resolution (um per pixel) for (x,y) axes. If none available, return (1.,1.)."""
        ...
    
    @property
    @abstractmethod
    def interval(self) -> float | None:
        """Return the time interval between frames in seconds, or None if not available."""
        ...
    
    @property
    @abstractmethod
    def custom_metadata(self) -> Mapping[str, Any]:
        """Return custom metadata, if any, associated with the custom pipeline saved under extratags, or empty dict if not available."""
        ...
    
    @abstractmethod
    def get_array(self) -> NDArray | list[NDArray]:
        """Return the image data as a NumPy array. If multiple series, return a list of arrays."""
        ...
    
    @abstractmethod
    def get_channel(self, channel: int | str | Sequence[int | str]) -> NDArray:
        """Return the selected channel(s) as a NumPy array. Channel can be specified by index or label."""
        ...

@dataclass(slots=True)
class Nd2Reader(ImageReader):
    
    _sizes: Mapping[str, int] = field(init=False)
    _axes: str = field(init=False)
    _channels: list[Channel] | None = field(init=False)
    _exploop: list[ExpLoop] = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() == '.nd2'

    def __post_init__(self) -> None:
        with nd2.ND2File(self.img_path) as file:
            self._sizes = file.sizes
            self._axes = ''.join(self._sizes.keys())
            meta = file.metadata
            self._channels = getattr(meta, 'channels', None)
            self._exploop = file.experiment
    
    @property
    def axes(self) -> str:
        return self._axes.replace('P', '')  # Remove 'P' axis for series handling.
    
    @property
    def status(self) -> str:
        return 'active'  # nd2 files do not have status info; default to 'active'
    
    @property
    def export_status(self,) -> str:
        return DEFAULT_FLAG
    
    @property
    def channel_number(self) -> int:
        return self._sizes.get('C', 1)
    
    @property
    def channel_labels(self) -> list[str] | None:
        if self._channels is None:
            return None
        
        labels: list[str] = []
        for channel in self._channels:
            chan: ChannelMeta = channel.channel
            labels.append(chan.name)
        return labels
    
    @property
    def series_number(self) -> int:
        return self._sizes.get('P', 1)
    
    @property
    def serie_axis_index(self) -> int | None:
        if 'P' not in self._axes:
            return None
        return self._axes.index('P')
            
    @property
    def resolution(self) -> PixelSize | None:
        if self._channels is None:
            return None
        
        ch0 = self._channels[0]
        vol: Volume | None = getattr(ch0, "volume", None)
        if vol is None:
            return None

        # (x, y, z)
        calib: tuple[float, float, float] | None = getattr(vol, "axesCalibration", None)
        if not calib:
            return None

        x_um_per_pix, y_um_per_pix = calib[:2]
        return (round(float(x_um_per_pix), 4), round(float(y_um_per_pix), 4))
        
    @property
    def interval(self) -> float | None:
        if self._sizes.get('T', 1) <= 1 or not self._exploop:
            return None
        
        loop0 = self._exploop[0]
        if loop0.type == 'TimeLoop': #for normal timelapse experiments
            return round(loop0.parameters.periodMs/1000)
        elif loop0.type == 'NETimeLoop': #for ND2 Merged Experiments
            return round(loop0.parameters.periods[0].periodMs/1000)
    
    @property
    def custom_metadata(self) -> Mapping[str, Any]:
        logger.info(".nd2 file do not have custom metadata saved")
        return {}
    
    def get_array(self) -> NDArray | list[NDArray]:
        arr = nd2.imread(self.img_path)
        
        if self.serie_axis_index is None:
            return arr

        series_lst = np.split(arr, arr.shape[self.serie_axis_index], axis=self.serie_axis_index)
        return [s.squeeze(axis=self.serie_axis_index) for s in series_lst]
    
    def get_channel(self, channel: int | str | Sequence[int | str]) -> NDArray:
        logger.info("Reading channel(s) from .nd2 file is not yet implemented")
        raise NotImplementedError("Channel reading from .nd2 files is not yet implemented")
    
@dataclass
class TiffReader(ImageReader):
    
    _tifserie0: TiffPageSeries = field(init=False)
    _axes: str = field(init=False)
    _page0: TiffPage = field(init=False)
    _imageJ_meta: dict[str, Any] = field(init=False)
    _status: StatusFlag = field(init=False)
    _custom_metadata: Mapping[str, Any] | None = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff']

    def __post_init__(self) -> None:
        with TiffFile(self.img_path) as tif:
            self._tifserie0 = tif.series[0]
            self._axes = self._tifserie0.axes
            self._page0 = cast(TiffPage, tif.pages[0]) # for typing purposes
            self._imageJ_meta = tif.imagej_metadata or {}
            
            meta = self._page0.tags.get(FITS_TAG)
            if meta is None:
                self._custom_metadata = None
            else:
                v = meta.value
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", "replace")
                try:
                    self._custom_metadata = json.loads(v)
                except Exception:
                    logger.warning("FITS_TAG present but not valid JSON")
                    self._custom_metadata = None
            
        self._status = self._get_status_from_metadata()
    
    def _get_status_from_metadata(self) -> StatusFlag:
        info = self._imageJ_meta.get("Info")
        prefix = "fits_io.status: "

        if not info or not info.startswith(prefix) or not isinstance(info, str):
            return DEFAULT_FLAG

        flag = info[len(prefix):].strip()
        return flag if flag in ALLOWED_FLAGS else DEFAULT_FLAG
     
    @property
    def axes(self) -> str:
        return self._axes.replace('S', '')  # Remove 'S' axis for series handling.
    
    @property
    def status(self) -> StatusFlag:
        return self._status
    
    @property
    def export_status(self,) -> str:
        return f"fits_io.status: {self._status}\n"
    
    @property
    def channel_number(self) -> int:
        shape = self._tifserie0.shape
        if 'C' in self._axes:
            c_index = self._axes.index('C')
            return shape[c_index]
        else:
            return 1
    
    @property
    def channel_labels(self) -> list[str] | None:
        labels = self._imageJ_meta.get('Labels', None)
        if labels is None:
            return None
        return labels
    
    @property
    def series_number(self) -> int:
        if 'S' in self._axes:
            s_index = self._axes.index('S')
            return self._tifserie0.shape[s_index]
        else:
            return 1
    
    @property
    def serie_axis_index(self) -> int | None:
        if 'S' not in self._axes:
            return None
        return self._axes.index('S')
    
    @property
    def resolution(self) -> PixelSize | None:
        x_res = self._page0.tags.get('XResolution')
        y_res = self._page0.tags.get('YResolution')
        
        if x_res is None or y_res is None:
            return None
        
        xres = x_res.value[0]/x_res.value[1]
        x_um_per_pix = round(1./float(xres), 4)
        yres = y_res.value[0]/y_res.value[1]
        y_um_per_pix = round(1./float(yres), 4)
        return (x_um_per_pix, y_um_per_pix)
                
    @property
    def interval(self) -> float | None:
        return self._imageJ_meta.get('finterval', None)
    
    @property
    def custom_metadata(self) -> Mapping[str, Any]:
        if self._custom_metadata is None:
            return {}
        return self._custom_metadata
    
    def get_array(self) -> NDArray | list[NDArray]:
        arr = imread(self.img_path)
        if self.serie_axis_index is None:
            return arr
        series_lst = np.split(arr, arr.shape[self.serie_axis_index], axis=self.serie_axis_index)
        return [s.squeeze(axis=self.serie_axis_index) for s in series_lst]

    def get_channel(self, channel: int | str | Sequence[int | str]) -> NDArray:
        if "S" in self._axes:
            raise NotImplementedError("Channel reading from multi-series TIFF files is not yet implemented")
        return read_tiff_channels(
            self.img_path,
            channel,
            channel_labels=self.channel_labels,
        )
    
class ImageReaderError(Exception):
    """Base exception for reader-related errors."""


class ReaderFileNotFoundError(ImageReaderError):
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
        raise ReaderFileNotFoundError(f"Path not found or is not a file: {p}")
    
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
    
    tif_path = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif')
    
    
    rtif = get_reader(tif_path)
    # print(rtif.channel_number)
    # print(rtif.serie_axis_index)
    print(rtif.resolution)
    # print(rtif.interval)
    # print(rtif.axes)
    # arrs = rtif.get_array()
    # if isinstance(arrs, list):
    #     print(len(arrs))
    #     print(arrs[0].shape)
    # else:
    #     print(arrs.shape)
        
    
    
    # nd2_path = Path('/home/ben/Docker_mount/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2')
    # save_path = Path('/home/ben/Docker_mount/Test_images/nd2/Run2/test.tiff')
    
    
    # with nd2.ND2File(nd2_path) as file:
    #     meta = file.metadata
    #     chans = getattr(meta, 'channels', None)
    #     if chans is not None:
    #         for ch in chans:
    #             print(ch.channel.name)
        # rnd2 = get_reader(nd2_path)
    # # # rnd2 = Nd2Reader(nd2_path)
    # print(rnd2.resolution)
    # print(rnd2.interval)
    # print(rnd2.axes)
    # print(rnd2.serie_axis_index)
    # print(len(rnd2.get_array()))
    