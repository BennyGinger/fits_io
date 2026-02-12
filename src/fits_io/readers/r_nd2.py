from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from pathlib import Path
import logging

import nd2
from nd2.structures import Channel, ExpLoop, ChannelMeta, Volume
import numpy as np
from numpy.typing import NDArray

from fits_io.readers.protocol import DEFAULT_FLAG, ImageReader
from fits_io.readers._types import PixelSize, StatusFlag, Zproj, ArrAxis
from fits_io.readers.info import InfoProfile


logger = logging.getLogger(__name__)

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
        
        self._validate_channel_label_override()
        if self._channel_labels is None:
            logger.info("Nd2Reader: channel_labels not provided; label-based selection will be disabled.")
    
    @property
    def axes(self) -> list[str]:
        return [self._axes.replace('P', '')]
    
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
        n_series = self.series_number
        n_channels = self._sizes.get("C", 1)
        return [n_channels] * n_series
    
    def _native_channel_labels(self) -> list[str] | None:
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
    
    def axis_index(self, axis: ArrAxis) -> list[int | None]:
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
            return self.apply_z_projection(arr, z_axis=z_axis, method=z_projection)

        series_lst = np.split(arr, arr.shape[p_axis], axis=p_axis)
        arr_lst = [s.squeeze(axis=p_axis) for s in series_lst]
        return [self.apply_z_projection(a, z_axis=z_axis, method=z_projection) for a in arr_lst]
    
    def _normalize_channels(self, channel: int | str | Sequence[int | str]) -> list[int]:
        req = [channel] if isinstance(channel, (int, str)) else list(channel)

        # determine n_channels
        n_channels = self.channel_number[0]

        out: list[int] = []
        for item in req:
            if isinstance(item, int):
                if not (0 <= item < n_channels):
                    raise IndexError(f"Channel index {item} out of range [0, {n_channels})")
                out.append(item)
            else:
                labels = self._channel_labels
                if labels is None:
                    raise ValueError(
                        f"Channel {item!r} requested by label, but {type(self).__name__} "
                "does not support native channel labels. Provide channel_labels in get_reader(...), "
                "or use integer channel indices."
            )
                try:
                    out.append(labels.index(item))
                except ValueError:
                    raise ValueError(f"Unknown channel label {item!r}. Known: {labels}") from None
        return out
    
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        # Get the different indexes and axes
        z_axis = self.axis_index('Z')[0]
        c_axis = self.axis_index('C')[0]
        p_axis = self.axis_index('P')[0]
        idxs = self._normalize_channels(channel)
        chan_idxs = idxs[0] if len(idxs) == 1 else idxs  # single int if only one channel requested, else list of ints
        
        # get dask array for lazy loading and channel selection
        darr = nd2.imread(self.img_path, dask=True)
        
        # Create the slicer
        slicer = [slice(None)] * darr.ndim
        slicer[c_axis] = chan_idxs
        
        chan_arr = darr[tuple(slicer)].compute()
        
        if p_axis is None:
            return self.apply_z_projection(chan_arr, z_axis=z_axis, method=z_projection)
        
        series_lst = np.split(chan_arr, chan_arr.shape[p_axis], axis=p_axis)
        arr_lst = [s.squeeze(axis=p_axis) for s in series_lst]
        return [self.apply_z_projection(a, z_axis=z_axis, method=z_projection) for a in arr_lst]