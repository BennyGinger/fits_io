from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar
from collections.abc import  Mapping, Sequence

from numpy.typing import NDArray
import numpy as np

from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.payload import build_payload
from fits_io.readers.protocol import ImageReader
from fits_io.readers._types import Zproj
from fits_io.readers.factory import get_reader
from fits_io.writers.apis import apply_zproj, convert_to_fits_tif, save_array, set_channel_labels, DEFAULT_OUTPUT_NAME


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
    def artefact_channel_indices(self) -> list[int] | None:
        """
        Returns the artifact channel indices from the metadata, if available.
        """
        return self.metadata.fits_io.artifact_channel_indices
    
    
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
    
    
    def get_array(self, z_projection: Zproj = None) -> NDArray[Any]:
        """
        Returns the image data NumPy array.
        Args:
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        """
        return self.reader.get_array(z_projection=z_projection)
    
    
    def get_channel(self, channel: int | str | Sequence[int | str], z_projection: Zproj = None) -> NDArray[Any]:
        """
        Returns the image data for a specific channel(s) as a NumPy array.
        Args:
            channel : Channel selector(s): int indices and/or str labels (all must be same type).
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
        """
        chan_idx = self._resolve_channel_indices(channel)
        return self.reader.get_channel(chan_idx, z_projection=z_projection)
    
    
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
    
    
    def convert_to_fits(self, 
                        *, 
                        output_name: str = DEFAULT_OUTPUT_NAME, 
                        channel_labels: str | Sequence[str] | None = None,
                        export_channels: str | Sequence[str] = 'all',
                        artifact_type: str = "image",
                        custom_metadata: Mapping[str, Any] | None = None, 
                        z_projection: Zproj = None, 
                        compression: str | None = 'zlib'
                        ) -> list[Path]:
        """
        Convert an image file to a FITS TIFF with ImageJ metadata. Supported input formats depend on installed image readers.
        Args:
            output_name : Optional name of the output TIFF file.
            channel_labels : Optional labels for source channels (used for mapping), if None, default labels will be used. 
            export_channels : Subset channels to export. Can be 'all' or a list of channel labels, by default 'all'.
            artifact_type : Type of artifact to set in the metadata, by default "image".
            custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
        Returns:
            List of Paths of the saved TIFF files.
        """
        save_paths = convert_to_fits_tif(self.reader,
                            output_name=output_name,
                            channel_labels=channel_labels, 
                            export_channels=export_channels,
                            artifact_type=artifact_type,
                            custom_metadata=custom_metadata,
                            z_projection=z_projection, 
                            compression=compression)
        return save_paths


    def save_array(self, 
                   array: NDArray, 
                   *, 
                   output_name: str = DEFAULT_OUTPUT_NAME, 
                   export_channels: str | Sequence[str],
                   artifact_type: str | None = None,
                   created_by: str | None = None,
                   z_projection: Zproj = None,
                   custom_metadata: Mapping[str, Any] | None = None,
                   compression: str | None = 'zlib',
                   ) -> Path:
        """
        Save the given array to a FITS TIFF file with ImageJ metadata.
        
        Policy:
        - This function will only save the provided array to a new file with the specified output name.
        - The metadata will be retrieved from the image reader, except for any custom metadata provided, which will be added or merged to existing metadata.
        - Multi-series inputs are not supported here by design.
        
        Args:
            array : The NumPy array to save.
            output_name : Optional name of the output TIFF file.
            export_channels : Subset channels to export. Can be 'all' or a list of channel labels.
            artifact_type : Type of artifact to set in the metadata, by default None.
            created_by : Optional string to set as the creator in the metadata (e.g. distributor), by default None.
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
            compression : Compression method to use for the TIFF file. If None, no compression is applied, by default 'zlib'.
            
        Returns:
            Path of the saved TIFF file.
        """
        
        meta = self._build_payload(export_channels=export_channels,
                                   artifact_type=artifact_type,
                                   created_by=created_by,
                                   z_projection=z_projection,
                                   custom_metadata=custom_metadata,
                                   array_shape=array.shape)
        
        return save_array(self.reader, 
                          array,
                          fitsio_metadata=meta,
                          output_name=output_name,
                          compression=compression)
    
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


    def _resolve_channel_indices(self, channels: int | str | Sequence[int | str]) -> list[int]:
        """
        Resolve channel indices from the provided channel selector(s) to always return a list of integer indices. 
        """
        labels = self.channel_labels

        if isinstance(channels, (int, str)):
            channels = [channels]

        mapping = {lbl: idx for idx, lbl in enumerate(labels)}
        indices: list[int] = []
        for ch in channels:
            if isinstance(ch, int):
                indices.append(ch)
            elif isinstance(ch, str):
                try:
                    indices.append(mapping[ch])
                except KeyError:
                    raise KeyError(f"Unknown channel label {ch!r}. Available labels: {labels}")
            else:
                raise TypeError(f"Expected int or str channel, got {type(ch).__name__}")
        return indices
    
    
    def _build_payload(self,
                      *,
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
            export_channels : Subset channels to export. Can be 'all' or a list of channel labels.
            artifact_type : Type of artifact to set in the metadata, by default None.
            created_by : Optional string to set as the creator in the metadata (e.g. distributor), by default None.
            z_projection : Z-projection method to apply ('max', 'mean', or None), by default None.
            custom_metadata : Additional custom metadata to include in the TIFF file, by default None.
            array_shape : Optional shape of the array to validate against the channel selection. If None, no validation is performed.
        
        Returns:
            FitsIOMeta: The constructed FITS I/O metadata payload.
        """
        
        return build_payload(self.reader,
                             channel_labels=self.channel_labels,
                             export_channels=export_channels,
                             artifact_type=artifact_type,
                             created_by=created_by,
                             z_projection=z_projection,
                             custom_metadata=custom_metadata,
                             array_shape=array_shape)


if __name__ == "__main__":
    path = Path("/media/ben/Analysis/Python/Images/nd2/Run2/c2z25t23v1_nd2.nd2")
    
    reader = FitsIO.from_path(path)
    file_path = reader.convert_to_fits(output_name="test_output.tif")
    
    read2 = FitsIO.from_path(file_path[0])
    read2.set_channel_labels(["RFP", "yellow"])
    red = read2.get_channel("yellow")
    
    path2 = read2.save_array(red, output_name="red_channel.tif", export_channels="yellow", custom_metadata={"note": "red channel only"})
    
    input("Press Enter to continue...")
    read3 = FitsIO.from_path(path2)
    read3.apply_z_projection("max", compression=None)