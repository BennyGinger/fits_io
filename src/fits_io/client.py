from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence

from numpy.typing import NDArray

from fits_io.image_reader import ImageReader, get_reader
from fits_io.writer import get_save_dirs, convert_to_fits_tif, save_fits_array
from fits_io.provenance import create_export_profile

ALLOWED_FILENAMES = {'fits_array.tif', 'fits_mask.tif'}


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
    
    def get_array(self) -> NDArray | list[NDArray]:
        """
        Returns the image data as a NumPy array or a list of arrays for multi-series files.
        """
        return self.reader.get_array()
    
    def get_channel_array(self, channel: int | str | Sequence[int | str]) -> NDArray:
        """
        Returns the image data for a specific channel(s) as a NumPy array
        """
        return self.reader.get_channel(channel)
    
    def get_save_dirs(self) -> Path | list[Path]:
        """
        Get the output directory paths name to save experiment converted arrays.
        """
        return get_save_dirs(self.reader)
    
    def convert_to_fits(self, *, channel_labels: str | Sequence[str] | None = None, user_defined_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
        """
        Convert an image file to a TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
        Args:
            channel_labels : Channel labels to include in the metadata. If None, generic labels will be created, by default None
            user_defined_metadata : Additional custom metadata to include in the TIFF file, by default None
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'
            overwrite : If True, overwrite existing files. If False and the output file exists, skip conversion, by default False
        """
        convert_to_fits_tif(self.reader, channel_labels=channel_labels, user_defined_metadata=user_defined_metadata, compression=compression, overwrite=overwrite)

    def save_fits_array(self, distribution: str | None = None, step_name: str | None = None, filename: str | None = None, user_metadata: Mapping[str, Any] | None = None, compression: str | None = 'zlib', overwrite: bool = False) -> None:
        """
        Save the FITS array to a TIFF file with ImageJ metadata.
        
        Policy:
        - distribution, step_name, filename are optional, in order to provide custom provenance tracking. If not provided, default values will be used.
        
        Args:
            distribution : Optional name of the distribution or package for provenance tracking.
            step_name : Optional name of the processing step for provenance tracking.
            filename : Optional name of the output TIFF file. Must be one of {'fits_array.tif', 'fits_mask.tif'}.
            user_metadata : Additional custom metadata to include in the TIFF file, by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'. Possible values are 'zlib', 'lzma', 'zstd', 'lz4', 'lzw', 'packbits' and 'jpeg'.
            overwrite : If True, overwrite existing files. If False and the output file exists, skip saving, by default False.
        """
        if filename not in ALLOWED_FILENAMES:
            raise ValueError(f"Invalid filename '{filename}'. Allowed filenames are: {ALLOWED_FILENAMES}")
        
        export_profile = create_export_profile(self.fits_metadata, distribution, step_name, filename)
        save_fits_array(self.reader, export_profile=export_profile, user_defined_metadata=user_metadata, compression=compression, overwrite=overwrite)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


