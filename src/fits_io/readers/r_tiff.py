from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, cast
from pathlib import Path
import json
import logging

from fits_io.metadata.provenance import FITS_TAG
from fits_io.readers.info import INFO_NAMESPACE, InfoProfile
from fits_io.readers.tiff_axis_io import read_tiff_channels
from numpy.typing import NDArray
from tifffile import TiffFile, TiffPage, imread, COMPRESSION, TiffTag

from fits_io.readers._types import ArrAxis, PixelSize, StatusFlag, Zproj
from fits_io.readers.protocol import ALLOWED_FLAGS, DEFAULT_FLAG, ImageReader


logger = logging.getLogger(__name__)

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
    _custom_metadata: Mapping[str, Any] = field(init=False)
    
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
        self._validate_channel_label_override()
    
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
    
    def _get_custom_metadata_from_tags(self, meta_tag: TiffTag | None) -> Mapping[str, Any]:
        if meta_tag is None:
            return {}
        
        v = meta_tag.value
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", "replace")
        try:
            return json.loads(v)
        except Exception:
            logger.warning("FITS_TAG present but not valid JSON")
            return {}
    
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
        
        flag = self._custom_metadata.get("status", DEFAULT_FLAG)
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
    def channel_number(self) -> list[int]:
        out: list[int] = []
        for i, shape in enumerate(self._shape):
            if 'C' in self._axes[i]:
                c_idx = self._axes[i].index('C')
                out.append(shape[c_idx])
            else:
                out.append(1)
        return out
    
    def _native_channel_labels(self) -> list[str] | None:
        labels = self._imageJ_meta.get('Labels', None)
        if isinstance(labels, str):
            return [labels.strip()]
        
        if isinstance(labels, list) and all(isinstance(lbl, str) for lbl in labels):
            return labels
        return None
    
    @property
    def series_number(self) -> int:
        return self._series_count
    
    def axis_index(self, axis: ArrAxis) -> list[int | None]:
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
        return self._custom_metadata
    
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        if self.series_number == 1:
            z_axis = self.axis_index('Z')[0]
            arr = imread(self.img_path)
            return self.apply_z_projection(arr, z_axis=z_axis, method=z_projection)
        
        with TiffFile(self.img_path) as tif:
            out: list[NDArray] = []
            for i, s in enumerate(tif.series):
                arr = s.asarray()
                z_axis = self.axis_index('Z')[i]
                z_arr = self.apply_z_projection(arr, z_axis=z_axis, method=z_projection)
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
            return self.apply_z_projection(chan_arr, z_axis=z_axis, method=z_projection)
        
        chan_arr_lst = [read_tiff_channels(
                self.img_path,
                channel,
                channel_labels=self.channel_labels,
                series_index=i,
            ) for i in range(self.series_number)]
        
        for i, arr in enumerate(chan_arr_lst):
            z_axis = self.axis_index('Z')[i]
            chan_arr_lst[i] = self.apply_z_projection(arr, z_axis=z_axis, method=z_projection)
        return chan_arr_lst