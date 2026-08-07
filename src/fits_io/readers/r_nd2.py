from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping, Sequence
from pathlib import Path
import logging

from fits_io.metadata.models import FitsIOMeta
import nd2
from nd2.structures import Channel, ExpLoop, Volume
import numpy as np
from numpy.typing import NDArray

from fits_io.readers.protocol import ImageReader
from fits_io.readers._types import PixelSize, Zproj, validate_axes


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Nd2Reader(ImageReader):
    # Inherited fields from ImageReader: 
        # img_path: Path
        # series_idx: int = 0
        # _shape: tuple[int, ...] = field(init=False)
        # _axes: str = field(init=False)
    
    # Additional fields specific to Nd2Reader
    _sizes: Mapping[str, int] = field(init=False)
    _channels: list[Channel] | None = field(init=False)
    _exploop: list[ExpLoop] = field(init=False)
    
    
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.suffix.lower() == '.nd2'

    def __post_init__(self) -> None:
        with nd2.ND2File(self.img_path) as file:
            self._sizes = file.sizes
            
            self._axes = ''.join(self._sizes.keys())
            axes = self._axes.replace('P', '') # Checking axes without P axis, since P axis is not present in the returned array
            validate_axes(axes)
            
            self._shape = file.shape
            
            meta = file.metadata
            self._channels = getattr(meta, 'channels', None)
            self._exploop = file.experiment
    
    @property
    def has_series(self) -> bool:
        return self._sizes.get('P', 1) > 1
    
    @property
    def series_count(self) -> int:
        return self._sizes.get('P', 1)
    
    def split_series(self) -> list[ImageReader]:
        if not self.has_series:
            return [self]
        
        return [Nd2Reader(self.img_path, series_idx=i) for i in range(self.series_count)]
    
    @property
    def axes(self) -> str:
        return self._axes.replace('P', '')
    
    @property
    def shape(self) -> tuple[int, ...]:
        p_idx = self._axes.find('P')
        if p_idx == -1:
            return self._shape
        return self._shape[:p_idx] + self._shape[p_idx+1:]
    
    @property
    def compression_method(self) -> str | None:
        return None  # nd2 files are never compressed
    
    @property
    def channel_count(self) -> int:
        return self._sizes.get("C", 1)
    
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
        if calib is None:
            return None

        x_um_per_pix, y_um_per_pix = calib[:2]
        value = (round(float(x_um_per_pix), 4), round(float(y_um_per_pix), 4))
        return value
        
    @property
    def interval(self) -> float | None:
        if self._sizes.get('T', 1) <= 1 or not self._exploop:
            return None
        
        for loop in self._exploop:
            match loop.type:
                case "TimeLoop":
                    return round(loop.parameters.periodMs / 1000)

                case "NETimeLoop":
                    return round(loop.parameters.periods[0].periodMs / 1000)
        return None
    
    @property
    def metadata(self) -> FitsIOMeta:
        # .nd2 file do not have custom metadata saved
        return FitsIOMeta()
    
    @property
    def zproj_method(self) -> Zproj:
        logger.warning(".nd2 files do not have z-projection method metadata; returning None")
        return None
    
    def get_array(self, z_projection: Zproj = None) -> NDArray[Any]:
        arr = nd2.imread(self.img_path)
        p_axis = self._axes.find('P')
        # Determine the Z axis index, if present. Note it will always give the position without the P axis, since the P axis is not present in the returned array.
        z_axis = self.axes.find('Z') if self.axes.find('Z') != -1 else None
        
        if p_axis == -1:
            return self.apply_zproj(arr, z_axis=z_axis, zproj=z_projection)

        # Remove the P axis by selecting the series index
        arr: NDArray[Any] = np.take(arr, indices=self.series_idx, axis=p_axis)
        return self.apply_zproj(arr, z_axis=z_axis, zproj=z_projection)
    
    def get_channel(self, channel: int | Sequence[int], z_projection: Zproj = None) -> NDArray[Any]:
        idxs = self._normalize_channel_indices(channel)
        self._validate_channel_indices(idxs)
        c_axis = self._axes.find('C')
        p_axis = self._axes.find('P')
        
        if c_axis == -1:
            if self.channel_count == 1 and len(idxs) == 1 and idxs[0] == 0:
                return self.get_array(z_projection)
            raise ValueError("Cannot select channels from ND2 data without a C axis.")
        
        # get dask array for lazy loading and channel selection
        darr = nd2.imread(self.img_path, dask=True)
        
        # Create the slicer
        chan_idxs = idxs[0] if len(idxs) == 1 else idxs  # single int if only one channel requested, else list of ints
        slicer = [slice(None)] * darr.ndim
        slicer[c_axis] = chan_idxs
        
        if p_axis != -1:
            slicer[p_axis] = self.series_idx

        chan_arr = darr[tuple(slicer)].compute()
        
        axes_after = self.axes
        if len(idxs) == 1:
            # If only one channel is selected, drop the C axis from the output
            axes_after = axes_after.replace('C', '')
        
        z_axis = axes_after.find("Z")
        z_axis = z_axis if z_axis != -1 else None
        return self.apply_zproj(chan_arr, z_axis=z_axis, zproj=z_projection)
    
    

if __name__ == "__main__":
    from pathlib import Path
    
    file_path = Path("/media/ben/Analysis/Python/Images/nd2/Run2/c2z25t23v1_nd2.nd2")
    
    reader = Nd2Reader(file_path)
    
    arr = reader.get_array(z_projection="max")
    print(f"Array shape: {arr.shape}")
    
    chan_arr = reader.get_channel(1, z_projection="max")
    print(f"Channel array shape: {chan_arr.shape}")
    
    print(reader.compression_method)
    print(reader.zproj_method)
    print(reader.has_series)
    print(reader.resolution)