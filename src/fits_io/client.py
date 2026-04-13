from pathlib import Path
from typing import Any, TypeVar
from collections.abc import  Mapping, Sequence

from numpy.typing import NDArray
import numpy as np

from fits_io.readers.protocol import ImageReader
from fits_io.readers._types import ArrAxis, ExtTags, Zproj
from fits_io.readers.factory import get_reader
from fits_io.writers.api import apply_zproj, convert_to_fits_tif, save_array, set_channel_labels, DEFAULT_OUTPUT_NAME
from fits_io.writers.filesystem import get_save_dirs


T = TypeVar('T', bound=np.generic)


class FitsIO:
    """
    Facade class for FITS I/O operations, providing simplified access to reading and converting FITS files.
    """
    def __init__(self, reader: ImageReader):
        self.reader = reader
        
    @classmethod
    def from_path(cls, path: str | Path, channel_labels: list[str] | None = None) -> 'FitsIO':
        reader = get_reader(path, channel_labels=channel_labels)
        return cls(reader)
    
    @property
    def axes(self) -> list[str]:
        """
        Returns the axis order as a list of strings e.g. ['TZCYX'].
        For multi-series files, returns one string per series.
        """
        return self.reader.axes
    
    @property
    def shape(self) -> list[tuple[int, ...]]:
        """
        Returns the shape of the image data for each series as a list of tuples.
        For multi-series files, returns one tuple per series.
        """
        return self.reader.shape
    
    def axis_index(self, axis: ArrAxis) -> list[int | None]:
        """
        Return the index of a given axis in the axis order string for each series (list).
        
        Args:
            axis: Single character representing the axis to find.
        Returns:
            List of indices of the axis for each series, or None if not present.
        """
        return self.reader.axis_index(axis)
    
    @property
    def fits_metadata(self) -> Mapping[str, Any]:
        """
        Returns the FITS metadata as a dictionary.
        """
        return self.reader.custom_metadata
    
    @property
    def channel_labels(self) -> list[str] | None:
        """
        Returns the channel labels from the metadata, or None if not available.
        """
        return self.reader.channel_labels
    
    def set_channel_labels(self, channel_labels: str | Sequence[str]) -> None:
        """
        Set the channel labels in the metadata.
        
        Policy:
        - This function will only change the channel labels in metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata.
        - Multi-series inputs are not supported here by design.
        
        Args:
            channel_labels : New channel labels to set, either a single string for one channel or a sequence of strings for multiple channels.
        """
        set_channel_labels(self.reader, channel_labels)
    
    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """
        Returns the image data as a NumPy array or a list of arrays for multi-series files.
        Args:
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        """
        return self.reader.get_array(z_projection=z_projection)
    
    def get_channel_array(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray[Any] | list[NDArray[Any]]:
        """
        Returns the image data for a specific channel(s) as a NumPy array or a list of arrays for multi-series files.
        Args:
            channel : Channel selector(s): int indices and/or str labels (all must be same type).
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        """
        return self.reader.get_channel(channel, z_projection=z_projection)
    
    def apply_z_projection(self, z_projection: Zproj | None) -> None:
        """
        Apply z-projection to the image array and update the file with the projected array and updated metadata.
    
        Policy:
        - This function will apply the specified z-projection to the existing array in the file and re-save it with updated metadata.
        - Multi-series inputs are not supported here by design.
        
        Args:
            img_reader : An ImageReader instance for the input image.
            z_projection : The z-projection method to apply ('max', 'mean', or None). If None, no projection is applied.
        """
        apply_zproj(self.reader, z_projection)
    
    def get_save_dirs(self) -> list[Path]:
        """
        Get the output directory paths name to save experiment converted arrays.
        """
        return get_save_dirs(self.reader)
    
    def convert_to_fits(self, *, channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', output_name: str = DEFAULT_OUTPUT_NAME, project_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None, compression: str | None = 'zlib') -> list[Path]:
        """
        Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
        Args:
            channel_labels : Channel labels to include in the metadata. If None, generic labels will be created, by default None
            export_channels : Channels to export. Can be 'all' or a list of channel labels, by default 'all'
            output_name : Optional name of the output TIFF file.
            project_metadata : Additional project-owned metadata to include in the TIFF file, by default None
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'.
        Returns:
            List of Paths of the saved TIFF files.
        """
        save_paths = convert_to_fits_tif(self.reader,
                            channel_labels=channel_labels, 
                            export_channels=export_channels,
                            output_name=output_name,
                            project_metadata=project_metadata,
                            z_projection=z_projection, 
                            compression=compression)
        return save_paths

    def save_array(self, array: NDArray, axis_order: str, channel_labels: str | Sequence[str] | None, output_name: str = DEFAULT_OUTPUT_NAME, *, project_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib') -> Path:
        """
        Save a given array as a FITS TIFF file with ImageJ metadata, using the input image's path as reference and transfer of metadata.
    
        Note: this function won't apply z-projection or channel label resolution. It simply saves the provided array with metadata built from reader context and parameters.
        
        Args:
            array : The image array to be saved as a TIFF file.
            output_name : Optional name of the output TIFF file. If None, uses 'fits.tif' by default.
            axis_order : Axis order of the input array, used for building correct metadata. Can raise an error in Tifffile if not provided correctly.
            channel_labels : Labels for the channels in the input array, used for building metadata. This should match the channels in the input array and can be a single string for one channel or a sequence of strings for multiple channels.
            project_metadata : Additional project-owned metadata to include in the TIFF file, by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default "zlib".
            
        Returns:
            Path of the saved TIFF file.
        """
        return save_array(self.reader, 
                          array,
                          axis_order=axis_order,
                          channel_labels=channel_labels,
                          output_name=output_name,
                          project_metadata=project_metadata,
                          compression=compression)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


