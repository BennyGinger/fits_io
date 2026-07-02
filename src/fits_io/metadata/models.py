from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Self, Sequence

from fits_io.readers._types import Zproj, validate_axes


@dataclass(slots=True, frozen=True)
class FitsIOMetadata:
    """
    Metadata container specific to fits_io.
    """
    version: str | None = None
    axes: str | None = None
    channel_labels: list[str] | None = None
    n_channels: int | None = None
    source_channel_indices: list[int] | None = None
    source_channel_count: int | None = None
    z_projection: Zproj = None
    compression: str | None = None

    def __post_init__(self) -> None:
        if self.axes is not None:
            validate_axes(self.axes)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "axes": self.axes,
            "channel_labels": self.channel_labels,
            "n_channels": self.n_channels,
            "source_channel_indices": self.source_channel_indices,
            "source_channel_count": self.source_channel_count,
            "z_projection": self.z_projection,
            "compression": self.compression,
        }


@dataclass(slots=True, frozen=True)
class FitsIOPayload:
    """
    Container for fits_io-specific metadata and any additional custom metadata. Represent the ground truth metadata for the image.
    """
    fits_io: FitsIOMetadata = field(default_factory=FitsIOMetadata)
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, payload_dict: Mapping[str, Any]) -> Self:
        """
        Create a FitsIOPayload instance from a dictionary representation.
        """
        fits_io_dict = payload_dict.get("fits_io", {})
        if not isinstance(fits_io_dict, Mapping):
            fits_io_dict = {}
    
        fits_io_meta = FitsIOMetadata(**fits_io_dict)
        
        custom_metadata = payload_dict.get("custom_metadata", {})
        if not isinstance(custom_metadata, Mapping):
            custom_metadata = {}
        return cls(fits_io=fits_io_meta, custom_metadata=custom_metadata)
    
    
    def dump(self) -> dict[str, Any]:
        """
        Convert the FitsIOPayload to a dictionary suitable for serialization or storage.
        """
        payload: dict[str, Any] = {}
        if self.fits_io is not None:
            payload['fits_io'] = self.fits_io.to_dict()
        if self.custom_metadata:
            payload["custom_metadata"] = dict(self.custom_metadata)
        return payload
    
    
    def to_info_payload(self) -> dict[str, Any]:
        """
        Convert the FitsIOPayload to a dictionary suitable for InfoSummary.
        """
        return {"fits_io": self.fits_io.to_dict(), **dict(self.custom_metadata),}
    
    
    def with_fitsio(self, 
                     *, 
                     version: str | None = None, 
                     axes: str | None = None, 
                     channel_labels: Sequence[str] | None = None, 
                     n_channels: int | None = None, 
                     source_channel_indices: Sequence[int] | None = None, 
                     source_channel_count: int | None = None, 
                     z_projection: str | None = None, 
                     compression: str | None = None
                     ) -> Self:
        """
        Return a new FitsIOPayload instance with updated fits_io metadata. If a parameter is None, the existing value is retained.
        """
        return replace(self,
                       fits_io=replace(self.fits_io,
                                       version=version if version is not None else self.fits_io.version,
                                       axes=axes if axes is not None else self.fits_io.axes,
                                       channel_labels=list(channel_labels) if channel_labels is not None else self.fits_io.channel_labels,
                                       n_channels=n_channels if n_channels is not None else self.fits_io.n_channels,
                                       source_channel_indices=list(source_channel_indices) if source_channel_indices is not None else self.fits_io.source_channel_indices,
                                       source_channel_count=source_channel_count if source_channel_count is not None else self.fits_io.source_channel_count,
                                       z_projection=z_projection if z_projection is not None else self.fits_io.z_projection,
                                       compression=compression if compression is not None else self.fits_io.compression,),
                       )
    
    def with_custom_metadata(self, custom_metadata: Mapping[str, Any] | None) -> Self:
        """
        Return a new FitsIOPayload instance with updated custom metadata. If custom_metadata is None, it will be set to an empty dictionary.
        """
        if custom_metadata is None:
            return self

        merged = dict(self.custom_metadata)
        merged.update(custom_metadata)
        return replace(self, custom_metadata=merged)
    










