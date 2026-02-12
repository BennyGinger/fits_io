from pathlib import Path
from typing import Any, Mapping, Sequence

from numpy.typing import NDArray

from fits_io.readers.protocol import ImageReader
from fits_io.readers._types import ExtTags, StatusFlag, Zproj
from fits_io.readers.factory import get_reader
from fits_io.writers.api import convert_to_fits_tif, save_fits_array, set_channel_labels, set_status, DEFAULT_OUTPUT_NAME
from fits_io.writers.filesystem import get_save_dirs


DISTRIBUTION_NAME = "fits-io"
STEP_NAME = "convert"
SUPPORTED_EXTENSIONS: set[ExtTags] = {'.tiff', '.tif', '.nd2'}


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
    def fits_metadata(self) -> Mapping[str, Any]:
        """
        Returns the FITS metadata as a dictionary.
        """
        return self.reader.custom_metadata
    
    @property
    def status(self) -> StatusFlag:
        """
        Returns the status of the image.
        """
        return self.reader.status
    
    def set_status(self, status: StatusFlag) -> None:
        """
        Set the status of the image to either 'active' or 'skip'.
        
        Policy:
        - This function will only change the status in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, channel labels, compression or provenance tag is applied here.
        - Multi-series inputs are not supported here by design.
        
        Args:
            status : New status to set ('active' or 'skip').
        """
        set_status(self.reader, status)
    
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
        - This function will only change the channel labels in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, change status, compression or provenance tag is applied here.
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
    
    def get_channel_array(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray | list[NDArray]:
        """
        Returns the image data for a specific channel(s) as a NumPy array or a list of arrays for multi-series files.
        Args:
            channel : Channel selector(s): int indices and/or str labels (all must be same type).
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        """
        return self.reader.get_channel(channel, z_projection=z_projection)
    
    def get_save_dirs(self) -> list[Path]:
        """
        Get the output directory paths name to save experiment converted arrays.
        """
        return get_save_dirs(self.reader)
    
    def convert_to_fits(self, *, user_name: str = 'unknown', channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', distribution: str | None = DISTRIBUTION_NAME, step_name: str | None = STEP_NAME, output_name: str = DEFAULT_OUTPUT_NAME, expected_filenames: set[str] | None = None, user_defined_metadata: Mapping[str, Any] | None = None, z_projection: Zproj = None,compression: str | None = 'zlib', overwrite: bool = False) -> list[Path]:
        """
        Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
        Args:
            user_name : Name of the user performing the conversion, by default 'unknown'
            channel_labels : Channel labels to include in the metadata. If None, generic labels will be created, by default None
            export_channels : Channels to export. Can be 'all' or a list of channel labels, by default 'all'
            distribution : Name of the distribution or package, by default None
            step_name : Name of the processing step, by default None
            output_name : Optional name of the output TIFF file.
            expected_filenames : Optional set of expected output filenames (without paths) to validate against after conversion, by default None
            user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'
            overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
        Returns:
            List of Paths of the saved TIFF files.
        """
        save_paths = convert_to_fits_tif(self.reader,
                            user_name=user_name,
                            channel_labels=channel_labels, 
                            export_channels=export_channels,
                            distribution=distribution, 
                            step_name=step_name, 
                            output_name=output_name,
                            expected_filenames=expected_filenames,
                            user_defined_metadata=user_defined_metadata,
                            z_projection=z_projection, 
                            compression=compression, 
                            overwrite=overwrite)
        return save_paths

    def save_fits_array(self, user_name: str = 'unknown', distribution: str | None = None, step_name: str | None = None, output_name: str = DEFAULT_OUTPUT_NAME, z_projection: Zproj = None, user_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib') -> None:
        """
        Save the FITS array to a TIFF file with ImageJ metadata.
        
        Policy:
        - distribution, step_name, output_name are optional, in order to provide custom provenance tracking. If not provided, default values will be used.
        
        Args:
            user_name : Name of the user performing the save operation, by default 'unknown'.
            distribution : Optional name of the distribution or package for provenance tracking.
            step_name : Optional name of the processing step for provenance tracking.
            output_name : Optional name of the output TIFF file. 
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            user_metadata : Additional custom metadata to include in the TIFF file, by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'.
        """
        save_fits_array(self.reader, 
                        user_name=user_name,
                        distribution=distribution, 
                        step_name=step_name, 
                        output_name=output_name,
                        user_defined_metadata=user_metadata, 
                        z_projection=z_projection, 
                        compression=compression)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


