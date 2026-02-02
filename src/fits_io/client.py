from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence

from numpy.typing import NDArray

from fits_io.image_reader import ImageReader, StatusFlag, get_reader, Zproj
from fits_io.writer import get_save_dirs, convert_to_fits_tif, save_fits_array, set_status
from fits_io.provenance import create_export_profile



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
        - This function will only change the status in the metadata, so it will load whatever array is already stored in the file and re-save it with updated metadata. So, no z-projection, channel selection or compression is applied here.
        - Multi-series inputs are not supported here by design.
        
        Args:
            status : New status to set ('active' or 'skip').
        """
        set_status(self.reader, status)
    
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
    
    def get_save_dirs(self) -> Path | list[Path]:
        """
        Get the output directory paths name to save experiment converted arrays.
        """
        return get_save_dirs(self.reader)
    
    def convert_to_fits(self, *, channel_labels: str | Sequence[str] | None = None, export_channels: str | Sequence[str] = 'all', distribution: str | None = None, step_name: str | None = None, filename: str | None = None, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
        """
        Convert an image file to a TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
        Args:
            channel_labels : Channel labels to include in the metadata. If None, generic labels will be created, by default None
            export_channels : Channels to export. Can be 'all' or a list of channel labels, by default 'all'
            distribution : Optional name of the distribution or package for provenance tracking.
            step_name : Optional name of the processing step for provenance tracking.
            filename : Optional name of the output TIFF file.
            user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'
            overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
        """
        export_profile = create_export_profile(self.fits_metadata, distribution, step_name, filename)
        convert_to_fits_tif(self.reader, channel_labels=channel_labels, export_channels=export_channels,export_profile=export_profile,user_defined_metadata=user_defined_metadata, compression=compression, overwrite=overwrite)

    def save_fits_array(self, distribution: str | None = None, step_name: str | None = None, filename: str | None = None, user_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
        """
        Save the FITS array to a TIFF file with ImageJ metadata.
        
        Policy:
        - distribution, step_name, filename are optional, in order to provide custom provenance tracking. If not provided, default values will be used.
        
        Args:
            distribution : Optional name of the distribution or package for provenance tracking.
            step_name : Optional name of the processing step for provenance tracking.
            filename : Optional name of the output TIFF file. 
            user_metadata : Additional custom metadata to include in the TIFF file, by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'.
            overwrite : If True, overwrite existing files. If False and the output file exists, skip saving, by default False.
        """
        export_profile = create_export_profile(self.fits_metadata, distribution, step_name, filename)
        save_fits_array(self.reader, export_profile=export_profile, user_defined_metadata=user_metadata, compression=compression, overwrite=overwrite)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


