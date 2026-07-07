from dataclasses import dataclass, field
from typing import Any, cast
from collections.abc import Sequence
from pathlib import Path
import json
import logging

from fits_io.metadata.codec import FITS_TAG
from fits_io.metadata.models import FitsIOPayload
import numpy as np
from numpy.typing import NDArray
from tifffile import TiffFile, TiffPage, TiffPageSeries, COMPRESSION, TiffTag

from fits_io.readers._types import PixelSize, Zproj, validate_axes
from fits_io.readers.protocol import ImageReader


logger = logging.getLogger(__name__)

@dataclass
class TiffReader(ImageReader):
    # Inherited fields from ImageReader: 
        # img_path: Path
        # series_idx: int = 0
        # _shape: tuple[int, ...] = field(init=False)
        # _axes: str = field(init=False)
    
    # Additional fields specific to TiffReader
    _series_list: list[TiffPageSeries] = field(init=False)
    _compression_method: str | None = field(init=False)
    _resolution: PixelSize | None = field(init=False)
    _imageJ_meta: dict[str, Any] = field(init=False)
    _metadata: FitsIOPayload = field(init=False)
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff']

    def __post_init__(self) -> None:
        with TiffFile(self.img_path) as tif:
            self._series_list = tif.series
            series = self._series_list[self.series_idx]
            self._shape = series.shape
            self._axes = series.axes
            validate_axes(self._axes)
            
            self._resolution = self._get_resolution_from_tags(cast(TiffPage, series.pages[0]))
            
            self._imageJ_meta = tif.imagej_metadata or {}
            
            self._compression_method = self._get_compression_from_tags(cast(TiffPage, series.pages[0]))
            
            meta = cast(TiffPage, series.pages[0]).tags.get(FITS_TAG)
            self._metadata = self._get_metadata_from_tags(meta)
    
    @property
    def has_series(self) -> bool:
        return len(self._series_list) > 1
    
    @property
    def series_count(self) -> int:
        return len(self._series_list)
    
    def split_series(self) -> list[ImageReader]:
        if not self.has_series:
            return [self]
        
        return [TiffReader(self.img_path, series_idx=i) for i in range(self.series_count)]
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    @property
    def axes(self) -> str:
        return self._axes
    
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
    
    def _get_metadata_from_tags(self, meta_tag: TiffTag | None) -> FitsIOPayload:
        if meta_tag is None:
            return FitsIOPayload()
        
        v = meta_tag.value
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", "replace")
        try:
            return FitsIOPayload.from_dict(json.loads(v))
        except Exception:
            logger.warning("FITS_TAG present but not valid JSON")
            return FitsIOPayload()
    
    @property
    def compression_method(self) -> str | None:
        return self._compression_method 
    
    @property
    def zproj_method(self) -> Zproj:
        fits_meta = self._metadata.fits_io
        if fits_meta is None:
            return None
        else:
            return fits_meta.z_projection
    
    @property
    def channel_count(self) -> int:
        if 'C' in self._axes:
            c_idx = self._axes.index('C')
            return self._shape[c_idx]
        return 1
    
    @property
    def resolution(self) -> PixelSize | None:
        return self._resolution
                
    @property
    def interval(self) -> float | None:
        return self._imageJ_meta.get('finterval', None)
    
    @property
    def metadata(self) -> FitsIOPayload:
        return self._metadata
    
    def get_array(self, z_projection: Zproj = None) -> NDArray[Any]:
        with TiffFile(self.img_path) as tif:
            series = tif.series[self.series_idx]
            arr = series.asarray()
            
        z_axis = self.axes.find('Z') 
        z_axis = z_axis if z_axis != -1 else None
        return self.apply_zproj(arr, z_axis=z_axis, zproj=z_projection)

    def get_channel(self, channel: int | Sequence[int], z_projection: Zproj = None) -> NDArray[Any]:
        c_list = self._normalize_channel_indices(channel)
        self._validate_channel_indices(c_list)
        
        with TiffFile(self.img_path) as tif:
            if not (0 <= self.series_idx < len(tif.series)):
                raise IndexError(f"series_index {self.series_idx} out of range (0..{len(tif.series) - 1})")

            s = tif.series[self.series_idx]
            axes = s.axes
            shape = s.shape
            
            c_axis = axes.find("C")
            c_axis = c_axis if c_axis != -1 else None

            if c_axis is None:
                if len(c_list) == 1 and c_list[0] == 0:
                    arr = s.asarray()
                    z_axis = axes.find("Z")
                    z_axis = z_axis if z_axis != -1 else None
                    return self.apply_zproj(arr, z_axis=z_axis, zproj=z_projection)
                raise ValueError(f"No 'C' axis in TIFF axes={axes!r}")

            page_axes = [ax for ax in axes if ax not in ("Y", "X")]
            page_shape = [shape[axes.index(ax)] for ax in page_axes]

            idx_grids: list[np.ndarray] = []
            for ax, n in zip(page_axes, page_shape):
                if ax == "C":
                    idx_grids.append(np.array(c_list, dtype=np.int64))
                else:
                    idx_grids.append(np.arange(n, dtype=np.int64))

            mesh = np.meshgrid(*idx_grids, indexing="ij")
            multi_idx = [m.ravel() for m in mesh]
            page_indices = np.ravel_multi_index(multi_idx, dims=page_shape, order="C")

            planes = tif.asarray(key=page_indices.tolist())

            out_page_shape = [len(c_list) if ax == "C" else shape[axes.index(ax)]
                for ax in page_axes]

            y = shape[axes.index("Y")]
            x = shape[axes.index("X")]

            chan_arr = planes.reshape(*out_page_shape, y, x)

            if len(c_list) == 1:
                chan_arr = np.squeeze(chan_arr, axis=page_axes.index("C"))
        
        axes_after = self.axes
        if len(c_list) == 1:
            # If only one channel is selected, drop the C axis from the output
            axes_after = self.axes.replace('C', '')
        
        z_axis = axes_after.find('Z')
        z_axis = z_axis if z_axis != -1 else None
        return self.apply_zproj(chan_arr, z_axis=z_axis, zproj=z_projection)
    

    
if __name__ == "__main__":
    from pathlib import Path
    
    file_path = Path("/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_001_s1/fits_array.tif")
    
    reader = TiffReader(file_path)
    
    arr = reader.get_array(z_projection=None)
    print(f"Array shape: {arr.shape}")
    
    chan_arr = reader.get_channel([0,1], z_projection=None)
    print(f"Channel array shape: {chan_arr.shape}")
    
    print(reader.compression_method)
    print(reader.zproj_method)
    print(reader.has_series)
    print(reader.resolution)
    print(reader.metadata)