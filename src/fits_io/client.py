from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar
from collections.abc import  Mapping, Sequence

from numpy.typing import NDArray
import numpy as np

from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.payload import build_payload
from fits_io.metadata.resolve import resolve_channel_selection, resolve_output_axes
from fits_io.readers.protocol import ImageReader
from fits_io.readers._types import Zproj
from fits_io.readers.factory import get_reader
from fits_io.writers.apis import apply_zproj, merge_channel_arrays, prepare_conversion, save_array, set_channel_labels 
from fits_io.writers.models import ChannelMergeResult, ConversionPreparation, ArrayResult, ChannelSubset, ChannelSelection
from fits_io.writers.filesystem import create_save_path


DEFAULT_OUTPUT_NAME = 'fits.tif'


T = TypeVar('T', bound=np.generic)


class FitsIO:
    """
    Facade class for FITS I/O operations, providing simplified access to reading and converting FITS files.
    """
    def __init__(self, reader: ImageReader):
        self.reader = reader
        
    
    @classmethod
    def from_path(cls, path: str | Path) -> FitsIO:
        reader = get_reader(path)
        return cls(reader)
    
    
    @property
    def axes(self) -> str:
        """
        Returns the axis order as a strings e.g. 'TZCYX'.
        """
        return self.reader.axes
    
    
    @property
    def metadata(self) -> FitsIOMeta:
        """
        Returns the FITS metadata.
        """
        return self.reader.metadata
    
    
    @property
    def channel_labels(self) -> list[str]:
        """
        Returns the channel labels from the metadata.
        
        Note: If none are found, it returns the default labels C_1, C_2, ..., C_n.
        """
        labels = self.metadata.fits_io.channel_labels
        if labels is None:
            n_channels = self.reader.channel_count
            labels = [f"C_{i+1}" for i in range(n_channels)]
        return labels
    
    
    @property
    def interval(self) -> float | None:
        """Return the frame interval in seconds."""
        return self.reader.interval
    
    
    @property
    def artifact_channel_indices(self) -> list[int]:
        """
        Returns the current source indices of the artifacts.
        """
        return self.metadata.fits_io.artifact_channel_indices
    
    
    @property
    def source_channel_indices(self) -> list[int]:
        """
        Returns the original source channel indices from the metadata, if available.
        """
        return self.metadata.fits_io.source_channel_indices
  
    
    def set_channel_labels(self, channel_labels: str | Sequence[str], compression: str | None = 'zlib') -> None:
        """
        Set the channel labels in the metadata.
        
        Policy:
        - This function will only change the channel labels in metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata.
        - Multi-series inputs are not supported here by design.
        
        Args:
            channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
        """
        path = set_channel_labels(self.reader, channel_labels, compression=compression)
        self.reader = get_reader(path)  # Reload the reader to reflect updated metadata.
    
    
    def get_array(self, z_projection: Zproj = None) -> ArrayResult:
        """
        Returns the image data NumPy array and the axes string from the current reader, optionally applying a z-projection.
        
        Args:
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        
        Returns:
            ArrayResult : A dataclass containing the NumPy array and the axes string.
        """
        array = self.reader.get_array(z_projection=z_projection)
        return ArrayResult(array=array, axes=self.reader.axes)
    
    
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> ArrayResult:
        """
        Returns the image data NumPy array for the specified channel(s) and the axes string from the current reader, optionally applying a z-projection.
        
        Args:
            channel : Channel selector(s): int indices and/or str labels (all must be same type).
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        
        Returns:
            ArrayResult : A dataclass containing the NumPy array and the axes string.
        """
        chan_positions = self.resolve_channel_positions(channel)
        out_axes = self._resolve_output_axes(z_projection=z_projection, n_channels=len(chan_positions),)
        array = self.reader.get_channel(chan_positions, z_projection=z_projection)
        
        return ArrayResult(array=array, axes=out_axes)
    
    
    def apply_z_projection(self, z_projection: Zproj | None, compression: str | None = 'zlib') -> None:
        """
        Apply z-projection to the image array and update the file with the projected array and updated metadata.
    
        Policy:
        - This function will apply the specified z-projection to the existing array in the file and re-save it with updated metadata.
        - Multi-series inputs are not supported here by design.
        
        Args:
            img_reader : An ImageReader instance for the input image.
            z_projection : The z-projection method to apply ('max', 'mean', or None). If None, no projection is applied.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
        """
        path = apply_zproj(self.reader, z_projection, compression=compression)
        self.reader = get_reader(path)  # Reload the reader to reflect updated metadata.
    
    
    def split_series(self) -> list[FitsIO]:
        """
        Split the current image into multiple series readers and return a list of FitsIO instances for each series.
        
        Returns:
            List of FitsIO instances, one for each series in the original image.
        """
        series_readers = self.reader.split_series()
        return [FitsIO(reader) for reader in series_readers]
    
    
    def labels_to_indices(self, requested_labels: str | Sequence[str],) -> list[int]:
        """
        Resolve channel labels to the source channel indices represented by the current artifact.
        """
        artifact_labels = self.channel_labels
        artifact_indices = self.artifact_channel_indices

        if artifact_indices is None:
            raise ValueError("Input artifact metadata is missing artifact_channel_indices.")

        if len(artifact_labels) != len(artifact_indices):
            raise ValueError("Input artifact metadata mismatch: "
                             f"channel_labels has {len(artifact_labels)} entries but "
                             f"artifact_channel_indices has {len(artifact_indices)}.")

        label_to_source = dict(zip(artifact_labels, artifact_indices, strict=True))

        try:
            if isinstance(requested_labels, str):
                requested_labels = [requested_labels]
            return [label_to_source[label] for label in requested_labels]
        except KeyError as exc:
            missing_label = exc.args[0]
            raise ValueError(f"Requested channel label {missing_label!r} was not found. "
                             f"Available labels: {artifact_labels}.") from exc
    
    
    def indices_to_labels(self, source_indices: int | Sequence[int],) -> list[str]:
        """
        Resolve source channel indices represented by the current artifact back to their channel labels.
        
        Duplicate source indices are ignored while preserving the first occurrence.
        """
        artifact_labels = self.channel_labels
        artifact_source_indices = self.artifact_channel_indices

        if artifact_source_indices is None:
            raise ValueError("Input artifact metadata is missing artifact_channel_indices.")

        if len(artifact_labels) != len(artifact_source_indices):
            raise ValueError("Input artifact metadata mismatch: "
                f"channel_labels has {len(artifact_labels)} entries but "
                f"artifact_channel_indices has {len(artifact_source_indices)}.")

        source_to_label = dict(zip(artifact_source_indices, artifact_labels, strict=True))

        if isinstance(source_indices, int):
            source_indices = [source_indices]

        source_indices = list(dict.fromkeys(source_indices)) # Remove duplicates while preserving order

        try:
            return [source_to_label[index] for index in source_indices]
        except KeyError as exc:
            missing_index = exc.args[0]
            raise ValueError(f"Source channel index {missing_index} was not found. "
                             f"Available source indices: {artifact_source_indices}.") from exc
    
    
    def resolve_channel_selection(self, 
                                  *,
                                  channel_labels: str | Sequence[str] | None = None, 
                                  export_channels: str | Sequence[str] = 'all'
                                  ) -> ChannelSelection:
        """
        Resolve channel indices and labels from the provided channel selector(s) and export channels. To ensure that updated metadata is saved in the output file, this function should be called before ``convert_to_fits`` or ``save_array``.
        
        Args:
            channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
            export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'.
        
        Returns:
            ChannelSelection : A dataclass containing the resolved channel indices and labels for export.
        """
        return resolve_channel_selection(channel_labels=channel_labels,
                                         n_channels=self.reader.channel_count,
                                         export_channels=export_channels)


    def resolve_channel_positions(self, channels: int | str | Sequence[int | str]) -> list[int]:
        """
        Resolve channel position from the provided channel selector(s) to always return a list of integer indices. 
        """
        if isinstance(channels, (int, str)):
            channels = [channels]
        
        channels = list(channels)
        if not channels:
            raise ValueError("At least one channel must be selected.")
        
        labels = self.channel_labels
        
        if 'C' not in self.axes:
            positions = self._resolve_singleton_positions(channels, labels)
        
        else:
            positions = self._resolve_multichannel_positions(channels, labels)
        
        if len(positions) != len(set(positions)):
            raise ValueError(f"Duplicate channel selection: {channels!r}.")
        return positions    

     
    def prepare_conversion(self,
                           selection: ChannelSelection,
                           *,
                           output_name: str = DEFAULT_OUTPUT_NAME,
                           artifact_kind: str | None = None,
                           created_by: str | None = None,
                           custom_metadata: Mapping[str, Any] | None = None,
                           z_projection: Zproj = None,
                           ) -> ConversionPreparation:
        """
        Prepare an array for initial conversion and return the selected array along with the output path.
        
        Returns:
            ConversionPreparation : A dataclass containing the selected array and the output path for saving.
        """
        return prepare_conversion(self.reader,
                                  selection=selection,
                                  output_name=output_name,
                                  artifact_kind=artifact_kind,
                                  created_by=created_by,
                                  custom_metadata=custom_metadata,
                                  z_projection=z_projection,)


    def save_array(self, 
                   array: NDArray, 
                   *,
                   channel_labels: str | Sequence[str] | None = None,
                   export_channels: str | Sequence[str] = 'all',
                   artifact_kind: str | None = None,
                   created_by: str | None = None,
                   z_projection: Zproj = None,
                   custom_metadata: Mapping[str, Any] | None = None, 
                   metadata: FitsIOMeta | None = None,
                   output_name: str = DEFAULT_OUTPUT_NAME, 
                   output_path: Path | None = None,
                   compression: str | None = 'zlib',
                   ) -> Path:
        """
        Save an array to a FITS TIFF.

        Policy:
            - By default, output metadata is built from the current reader and the
            provided artifact parameters. A fully constructed metadata object may
            instead be supplied through ``metadata``. If so, then ``export_channels``, ``artifact_type``, ``created_by``, ``z_projection``, and ``custom_metadata`` are ignored.
            - By default, the output path is constructed from the current reader's path and ``output_name``. A fully constructed path may instead be supplied through ``output_path``. If so, then ``output_name`` is ignored.
        
        Args:
            array : 
                The NumPy array to save.
            export_channels : 
                Subset channels to export. Can be 'all' or a list of channel labels.
            artifact_type : 
                Type of artifact to set in the metadata. If None, it will use the current type. By default None.
            created_by : 
                Optional string to set as the creator in the metadata (e.g. distributor). If None, it will use the current creator. By default None.
            z_projection : 
                Z-projection method to apply ('max', 'mean', or None). If None, it will use the current z-projection. By default None.
            custom_metadata : 
                Additional custom metadata to include in the TIFF file. If None, it will use the current custom metadata. By default None.
            metadata : 
                Optional pre-built FitsIOMeta object to use for saving. If provided, it will override the other metadata parameters.
            output_name : 
                Name of the output TIFF file. Ignored if ``output_path`` is provided. By default 'fits.tif'.
            output_path : 
                Optional full path for the output TIFF file. If provided, it will override ``output_name``.
            compression : 
                Compression method to use for the TIFF file. If None, no compression is applied. By default 'zlib'.
        
        Returns:
            Path : The path of the saved TIFF file.
        """
        if output_path is None:
            output_path = create_save_path(self.reader.img_path, output_name,)

        meta = metadata or self.build_payload(channel_labels=channel_labels,
                                              export_channels=export_channels,
                                              artifact_type=artifact_kind,
                                              created_by=created_by,
                                              z_projection=z_projection,
                                              custom_metadata=custom_metadata,
                                              array_shape=array.shape,)
        
        return save_array(self.reader,
                          array,
                          fitsio_metadata=meta,
                          output_path=output_path,
                          compression=compression,)
    
    
    def build_payload(self,
                      *,
                      channel_labels: str | Sequence[str] | None = None,
                      export_channels: str | Sequence[str],
                      artifact_type: str | None = None,
                      created_by: str | None = None,
                      z_projection: Zproj = None,
                      custom_metadata: Mapping[str, Any] | None = None,
                      array_shape: tuple[int, ...] | None = None,
                      ) -> FitsIOMeta:
        """
        Build a FITS I/O metadata payload based on the current reader and provided parameters.
        
        Args:
            export_channels : 
                Subset channels to export. Can be 'all' or a list of channel labels.
            artifact_type : 
                Type of artifact to set in the metadata, by default None.
            created_by : 
                Optional string to set as the creator in the metadata (e.g. distributor), by default None.
            z_projection : 
                Z-projection method to apply ('max', 'mean', or None), by default None.
            custom_metadata :   
                Additional custom metadata to include in the TIFF file, by default None.
            array_shape : 
                Optional shape of the array to validate against the channel selection. If None, no validation is performed.
        
        Returns:
            FitsIOMeta: The constructed FITS I/O metadata payload.
        """
        if channel_labels is None:
            channel_labels = self.channel_labels
        selection = self.resolve_channel_selection(channel_labels=channel_labels,
                                                   export_channels=export_channels)
        
        return build_payload(self.reader,
                             selection=selection,
                             artifact_kind=artifact_type,
                             created_by=created_by,
                             z_projection=z_projection,
                             custom_metadata=custom_metadata,
                             array_shape=array_shape,)


    def select_included_channels(self, excluded_labels: Sequence[str] | None,) -> ChannelSubset[Any]:
        """
        Load the channels that should be processed.

        Channel positions refer to positions along the current artifact's C axis,
        not source channel indices.

        No exclusions, no C axis, or exclusion of all available channels means
        that the complete array is selected.
        
        Returns:
            ChannelSubset : A dataclass containing the selected array and the channel positions for processing and the rebuild method to reconstruct the full array.
        """
        if not excluded_labels or "C" not in self.axes:
            return ChannelSubset(reader=self,
                                 array=self.get_array().array,
                                 channel_positions=(),)

        excluded_positions = set(self.resolve_channel_positions(excluded_labels))

        included_positions = tuple(position
                                   for position in range(self.reader.channel_count)
                                   if position not in excluded_positions)

        # Existing policy: excluding all channels means process the full array.
        if not included_positions:
            return ChannelSubset(reader=self,
                                 array=self.get_array().array,
                                 channel_positions=(),)

        return ChannelSubset(reader=self,
                             array=self.get_channel(included_positions).array,
                             channel_positions=included_positions,)


    def replace_channels(self,
                         channel: int | str | Sequence[int | str],
                         replacement: NDArray[Any],
                         ) -> NDArray[Any]:
        """
        Return the complete array with selected channels replaced.
        """
        chan_positions = self.resolve_channel_positions(channel)
        result = self.get_array()
        output = result.array
        
        channel_axis = self.axes.index("C")
        # Add a channel axis to the replacement if it is a single channel (missing C axis)
        if len(chan_positions) == 1 and replacement.ndim == output.ndim - 1:
            replacement = np.expand_dims(replacement,
                                         axis=channel_axis,)
        
        expected_shape = list(output.shape)
        expected_shape[channel_axis] = len(chan_positions)
        expected_shape = tuple(expected_shape)

        if replacement.shape != expected_shape:
            raise ValueError("Replacement shape does not match the selected channels: " + 
                             f"expected {expected_shape}, got {replacement.shape}.")

        output_view = np.moveaxis(output, channel_axis, 0)
        replacement_view = np.moveaxis(replacement, channel_axis, 0)
        output_view[chan_positions] = replacement_view

        return output
    
    
    def merge_channels(self,
                       *,
                       existing: FitsIO | None,
                       new_array: NDArray[Any],
                       new_axes: str,
                       new_channel_indices: Sequence[int],
                       ) -> ChannelMergeResult:
        new_indices = list(new_channel_indices)
        
        if existing is None:
            return ChannelMergeResult(array=new_array, axes=new_axes, channel_indices=new_indices)
        
        existing_indices = existing.artifact_channel_indices
        
        if existing_indices is None:
            raise ValueError(f"Existing artifact at {self.reader.img_path} "
                             "does not have artifact channel indices.")
    
        existing_results = existing.get_array()
        
        return merge_channel_arrays(existing_array=existing_results.array,
                                    existing_axes=existing_results.axes,
                                    existing_channel_indices=existing_indices,
                                    new_array=new_array,
                                    new_axes=new_axes,
                                    new_channel_indices=new_channel_indices,
                                    reference_axes=self.axes,)


    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


    def _resolve_output_axes(self, z_projection: Zproj = None, n_channels: int | None = None) -> str:
        """
        Resolve output axes string for ImageJ metadata based on the current reader's axes and provided parameters.
        
        Args:
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            n_channels : Optional number of channels to consider for output axes. If None, uses the current reader's channel count.
        
        Returns:
            str : The resolved output axes string.
        """
        if n_channels is None:
            n_channels = self.reader.channel_count
        return resolve_output_axes(reader_axes=self.reader.axes,
                                   z_projection=z_projection,
                                   n_channels=n_channels)
   
    def _resolve_singleton_positions(self, channels: Sequence[int | str], labels: Sequence[str]) -> list[int]:
        if len(labels) != 1:
            raise ValueError("No 'C' axis in the image, but multiple channel labels are present. Cannot resolve channel positions.")
    
        valid_label = labels[0]
        positions: list[int] = []
        
        for ch in channels:
            if isinstance(ch, int):
                if ch != 0:
                    raise ValueError(f"Channel index {ch} is out of range. This single-channel image only has channel index 0.")
                positions.append(0)
            
            elif isinstance(ch, str):
                if ch != valid_label:
                    raise ValueError(f"Channel label {ch!r} is not valid. This single-channel image only has channel label {valid_label!r}.")
                positions.append(0)
            
            else:
                raise TypeError(f"Expected int or str channel, got {type(ch).__name__}")
        
        return positions
    
    def _resolve_multichannel_positions(self, channels: Sequence[int | str], labels: Sequence[str]) -> list[int]:
        mapping = {label: position
                   for position, label in enumerate(labels)} 
                    
        positions = []
        
        for ch in channels:
            if isinstance(ch, int):
                if ch < 0 or ch >= len(labels):
                    raise ValueError(f"Channel index {ch} is out of range. "
                                    f"Available indices: 0 to {len(labels) - 1}.")
                positions.append(ch)
            
            elif isinstance(ch, str):
                try:
                    positions.append(mapping[ch])
                except KeyError as exc:
                    raise ValueError(f"Unknown channel label {ch!r}. Available labels: {labels}") from exc
            else:
                raise TypeError(f"Expected int or str channel, got {type(ch).__name__}")
        return positions

