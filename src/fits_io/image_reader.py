from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Type, cast
import logging

from fits_io.tiff_axis_io import apply_z_projection, read_tiff_channels
import nd2
from nd2.structures import Channel, ExpLoop, ChannelMeta, Volume
from tifffile import TiffFile, TiffPage, TiffWriter, imread, COMPRESSION, TiffTag
from numpy.typing import NDArray
import numpy as np

from fits_io.provenance import FITS_TAG
from fits_io._types import PixelSize

ExtTags = Literal['.tiff', '.tif', '.nd2']
SUPPORTED_EXTENSIONS: set[ExtTags] = {'.tiff', '.tif', '.nd2'}
StatusFlag = Literal["active", "skip"]
ALLOWED_FLAGS: set[StatusFlag] = {"active", "skip"}
DEFAULT_FLAG: StatusFlag = "active"
INFO_NAMESPACE = "fits_io"
Zproj = Literal['max', 'mean', None]

logger = logging.getLogger(__name__)

@dataclass
class InfoProfile:
    """Class to build the string for ImageJ Info metadata"""
    status: StatusFlag
    user: str = "unknown"
    namespace: str = INFO_NAMESPACE
    
    @property
    def _status(self) -> str:
        return f"{self.namespace}.status: {self.status}"
    
    @property
    def _user(self) -> str:
        return f"{self.namespace}.user: {self.user}"

    @property
    def export(self) -> str:
        lines = [
            self._status,
            self._user,
            ]
        return "\n".join(lines) + "\n"

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
    def axes(self) -> list[str]:
        """Return the axes string for the image data"""
        ...
    
    @property
    @abstractmethod
    def compression_method(self) -> str | None:
        """Return the compression method used for the image data, or None if uncompressed."""
        ...
    
    @property
    @abstractmethod
    def status(self) -> StatusFlag:
        """Return the status of the image for downstream processing (i.e., 'active' or 'skip')."""
        ...
    
    @property
    @abstractmethod
    def export_status(self,) -> str:
        """Return the export status string for the image."""
        ...
    
    @property
    @abstractmethod
    def channel_number(self) -> list[int]:
        """Return the number of channels in the image for each series, or 1 if not applicable."""
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
    
    @abstractmethod
    def axis_index(self, axis: Literal['P', 'C', 'Z', 'T', 'X', 'Y']) -> list[int | None]:
        """Return a list of indices of the specified axis, or None if not present, for each series."""
        ...
    
    @property
    @abstractmethod
    def resolution(self) -> list[PixelSize | None]:
        """Return a list of resolution (um per pixel) for (x,y) axes. If none available, return (1.,1.), for each series."""
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
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """Return the image data as a NumPy array. If multiple series, return a list of arrays."""
        ...
    
    @abstractmethod
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """Return the selected channel(s) as a NumPy array or list of arrays. Channel can be specified by index or label."""
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
    def axes(self) -> list[str]:
        return [self._axes.replace('P', '')]  # Remove 'P' axis for series handling.
    
    @property
    def compression_method(self) -> str | None:
        return None  # nd2 files are never compressed
    
    @property
    def status(self) -> StatusFlag:
        return 'active'  # nd2 files do not have status info; default to 'active'
    
    @property
    def export_status(self,) -> str:
        return InfoProfile(status=DEFAULT_FLAG).export
    
    @property
    def channel_number(self) -> list[int]:
        return [self._sizes.get('C', 1)]
    
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
    
    def axis_index(self, axis: Literal['P', 'C', 'Z', 'T', 'X', 'Y']) -> list[int | None]:
        if axis not in self._axes:
            return [None] * self.series_number
        return [self._axes.index(axis)] * self.series_number
            
    @property
    def resolution(self) -> list[PixelSize | None]:
        if self._channels is None:
            return [None]
        
        ch0 = self._channels[0]
        vol: Volume | None = getattr(ch0, "volume", None)
        if vol is None:
            return [None]

        # (x, y, z)
        calib: tuple[float, float, float] | None = getattr(vol, "axesCalibration", None)
        if not calib:
            return [None]

        x_um_per_pix, y_um_per_pix = calib[:2]
        return [(round(float(x_um_per_pix), 4), round(float(y_um_per_pix), 4))]
        
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
    
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        arr = nd2.imread(self.img_path)
        p_axis = self.axis_index('P')[0]
        z_axis = self.axis_index('Z')[0]
        
        if p_axis is None:
            return apply_z_projection(arr, z_axis=z_axis, method=z_projection)

        series_lst = np.split(arr, arr.shape[p_axis], axis=p_axis)
        arr_lst = [s.squeeze(axis=p_axis) for s in series_lst]
        return [apply_z_projection(a, z_axis=z_axis, method=z_projection) for a in arr_lst]
    
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        logger.warning("Reading channel(s) from .nd2 file is not yet implemented")
        return np.array([])
    
@dataclass
class TiffReader(ImageReader):
    
    _shape: list[tuple[int, ...]] = field(init=False)
    _axes: list[str] = field(init=False)
    # Series are handled as separate files internally, like a list of images, so no real axis
    _series_count: int = field(init=False)
    _compression_method: list[str | None] = field(init=False)
    _resolution: list[PixelSize | None] = field(init=False, default_factory=list)
    _imageJ_meta: dict[str, Any] = field(init=False)
    _status: StatusFlag = field(init=False)
    _custom_metadata: Mapping[str, Any] | None = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff']

    def __post_init__(self) -> None:
        with TiffFile(self.img_path) as tif:
            self._series_count = len(tif.series)
            
            self._shape = [s.shape for s in tif.series]
            self._axes = [s.axes for s in tif.series] # Only possible axes: C, Z, T, Y, X
            
            
            self._resolution = [self._get_resolution_from_tags(cast(TiffPage, s.pages[0]))
                    for s in tif.series]
            
            self._imageJ_meta = tif.imagej_metadata or {}
            
            self._compression_method = [self._get_compression_from_tags(cast(TiffPage, s.pages[0])) 
                    for s in tif.series]
            
            meta = cast(TiffPage, tif.series[0].pages[0]).tags.get(FITS_TAG)
            self._custom_metadata = self._get_custom_metadata_from_tags(meta)
            
        self._status = self._get_status_from_metadata()
    
    def _get_compression_from_tags(self, tiff_page: TiffPage) -> str | None:
        comp = COMPRESSION(tiff_page.tags["Compression"].value)
        return comp.name if comp != COMPRESSION.NONE else None
    
    def _get_resolution_from_tags(self, tiff_page: TiffPage) -> PixelSize | None:
        xres = tiff_page.tags.get('XResolution')
        yres = tiff_page.tags.get('YResolution')
        
        if xres is None or yres is None:
            return None
        
        xres = xres.value[0]/xres.value[1]
        x_um_per_pix = round(1./float(xres), 4)
        yres = yres.value[0]/yres.value[1]
        y_um_per_pix = round(1./float(yres), 4)
        return (x_um_per_pix, y_um_per_pix)
    
    def _get_custom_metadata_from_tags(self, meta_tag: TiffTag | None) -> Mapping[str, Any] | None:
        if meta_tag is None:
            return None
        
        v = meta_tag.value
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", "replace")
        try:
            return json.loads(v)
        except Exception:
            logger.warning("FITS_TAG present but not valid JSON")
            return None
    
    def _parse_info(self) -> dict[str, str]:
        r"""
        Convert ImageJ Info string to a dictionary. Expected format is 'key: value' per line, so the info output should be something like 'key: value\nkey2: value2\n...'.
        """
        info = self._imageJ_meta.get("Info")
        if not isinstance(info, str):
            return {}

        out: dict[str, str] = {}
        for line in info.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
        return out
    
    def _get_status_from_metadata(self) -> StatusFlag:
        info = self._parse_info()
        prefix = INFO_NAMESPACE

        flag = info.get(f"{prefix}.status", DEFAULT_FLAG)
        return flag if flag in ALLOWED_FLAGS else DEFAULT_FLAG
    
    @property
    def axes(self) -> list[str]:
        return self._axes
    
    @property
    def compression_method(self) -> str | None:
        # Not expecting multi-series with different compression when using this property
        return self._compression_method[0] 
    
    @property
    def status(self) -> StatusFlag:
        return self._status
    
    @property
    def export_status(self,) -> str:
        return InfoProfile(status=self._status).export
    
    @property
    def channel_number(self) -> list[int]:
        out: list[int] = []
        for i, shape in enumerate(self._shape):
            if 'C' in self._axes[i]:
                c_idx = self._axes[i].index('C')
                out.append(shape[c_idx])
            else:
                out.append(1)
        return out
    
    @property
    def channel_labels(self) -> list[str] | None:
        labels = self._imageJ_meta.get('Labels', None)
        if isinstance(labels, list) and all(isinstance(lbl, str) for lbl in labels):
            return labels
        return None
    
    @property
    def series_number(self) -> int:
        return self._series_count
    
    def axis_index(self, axis: Literal['P', 'C', 'Z', 'T', 'X', 'Y']) -> list[int | None]:
        out: list[int | None] = []
        for ax in self._axes:
            if axis not in ax:
                out.append(None)
            else:
                out.append(ax.index(axis))
        return out
    
    @property
    def resolution(self) -> list[PixelSize | None]:
        return self._resolution
                
    @property
    def interval(self) -> float | None:
        return self._imageJ_meta.get('finterval', None)
    
    @property
    def custom_metadata(self) -> Mapping[str, Any]:
        if self._custom_metadata is None:
            return {}
        return self._custom_metadata
    
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        if self.series_number == 1:
            z_axis = self.axis_index('Z')[0]
            arr = imread(self.img_path)
            return apply_z_projection(arr, z_axis=z_axis, method=z_projection)
        
        with TiffFile(self.img_path) as tif:
            out: list[NDArray] = []
            for i, s in enumerate(tif.series):
                arr = s.asarray()
                z_axis = self.axis_index('Z')[i]
                z_arr = apply_z_projection(arr, z_axis=z_axis, method=z_projection)
                out.append(z_arr)
        return out

    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        if self.series_number == 1:
            z_axis = self.axis_index('Z')[0]
            chan_arr = read_tiff_channels(
                self.img_path,
                channel,
                channel_labels=self.channel_labels,
                series_index=0,)
            return apply_z_projection(chan_arr, z_axis=z_axis, method=z_projection)
        
        chan_arr_lst = [read_tiff_channels(
                self.img_path,
                channel,
                channel_labels=self.channel_labels,
                series_index=i,
            ) for i in range(self.series_number)]
        
        for i, arr in enumerate(chan_arr_lst):
            z_axis = self.axis_index('Z')[i]
            chan_arr_lst[i] = apply_z_projection(arr, z_axis=z_axis, method=z_projection)
        return chan_arr_lst
    
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
    
    tif_path1 = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/c2z25t23v1_tif.tif')
    stack1 = imread(tif_path1)
    print(stack1.shape)
    tif_path2 = Path('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run4/c4z1t91v1.tif')
    stack2 = imread(tif_path2)
    print(stack2.shape)
    
    with TiffWriter('/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif') as tif:
        tif.write(stack1, metadata={'axes':'TZCYX'}, compression=None)
        tif.write(stack2, metadata={'axes':'TCYX'}, compression='lzw', photometric="minisblack", planarconfig="separate")
    
    with TiffFile("/media/ben/Analysis/Python/Docker_mount/Test_images/tiff/Run2/test.tif") as tif:
        print("n_series:", len(tif.series))
        for i, s in enumerate(tif.series):
            print(i, "axes:", s.axes, "shape:", s.shape, "n_pages:", len(s.pages))
    