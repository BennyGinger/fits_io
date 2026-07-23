from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, TYPE_CHECKING

from numpy.typing import NDArray
import numpy as np

if TYPE_CHECKING:
    from fits_io.client import FitsIO

from fits_io.readers._types import validate_axes
from fits_io.metadata.models import FitsIOMeta


T = TypeVar('T', bound=np.generic)


@dataclass(slots=True, frozen=True)
class ChannelSelection:
    """
    Represents the selection of channels for export, including the source labels, export labels, and export indices.
    
    Attributes:
        source_labels: The original channel labels from the reader.
        export_labels: The labels of the channels to be exported.
        export_indices: The indices of the channels to be exported, corresponding to the source labels.
    """
    source_labels: list[str]
    export_labels: list[str]
    export_indices: list[int]
    
    def validate_array(self, array_shape: tuple[int, ...], axes: str) -> None:
        """
        Validate that the provided array shape and axes match the expected channel selection.
        
        Args:
            array_shape: The shape of the array to validate.
            axes: The axes string corresponding to the array.
        """
        validate_axes(axes)

        if len(array_shape) != len(axes):
            raise ValueError(f"Array shape {array_shape} does not match axes '{axes}'.")

        c_index = axes.find('C')
        if c_index == -1:
            if len(self.export_indices) > 1:
                raise ValueError(f"Array has no channel axis, but export indices are {self.export_indices}.")
            else:
                return  # No channel axis and only one channel selected, valid case.
        c_axis_size = array_shape[c_index]
        if c_axis_size != len(self.export_indices):
            raise ValueError(f"Array channel axis size {c_axis_size} does not match number of export channels {len(self.export_indices)}.")


@dataclass(slots=True, frozen=True)
class ConversionPreparation:
    array: NDArray[Any]
    output_path: Path
    metadata: FitsIOMeta
    

@dataclass(slots=True, frozen=True)
class ArrayResult:
    array: NDArray[Any]
    axes: str
    

@dataclass(slots=True, frozen=True)
class ChannelSubset(Generic[T]):
    reader: FitsIO
    array: NDArray[T]
    channel_positions: tuple[int, ...]

    @property
    def is_full_array(self) -> bool:
        return not self.channel_positions

    @property
    def processed_indices(self) -> tuple[int, ...]:
        """
        Return the original source indices represented by the selected channels.
        """
        source_indices = self.reader.artifact_channel_indices

        if not self.channel_positions:
            return tuple(source_indices)

        return tuple(source_indices[position]
                     for position in self.channel_positions)
    
    def rebuild(self, processed_array: NDArray[T]) -> NDArray[T]:
        if processed_array.shape != self.array.shape:
            raise ValueError("Processed array shape does not match the selected input: "
                             f"expected {self.array.shape}, got {processed_array.shape}.")

        if self.is_full_array:
            return processed_array

        return self.reader.replace_channels(self.channel_positions, processed_array,)
    

@dataclass(slots=True, frozen=True)
class ChannelMergeResult:
    array: NDArray[Any]
    axes: str
    channel_indices: list[int]