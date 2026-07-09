from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Self, Sequence

from fits_io.readers._types import Zproj, validate_axes


@dataclass(slots=True, frozen=True)
class ArtifactMeta:
    """
    Metadata container specific to fits_io.
    """
    artifact_type: str | None = None
    created_by: str | None = None
    version: str | None = None
    derived_from: str | None = None
    axes: str | None = None
    channel_labels: list[str] | None = None
    source_channel_indices: list[int] | None = None
    artifact_channel_indices: list[int] | None = None
    z_projection: Zproj = None

    def __post_init__(self) -> None:
        if self.axes is not None:
            validate_axes(self.axes)
    
    @property
    def channel_count(self) -> int | None:
        """Return the number of channels in the artifact, if available."""
        if self.channel_labels is not None:
            return len(self.channel_labels)
        return None
    
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "created_by": self.created_by,
            "version": self.version,
            "derived_from": self.derived_from,
            "axes": self.axes,
            "channel_labels": self.channel_labels,
            "source_channel_indices": self.source_channel_indices,
            "artifact_channel_indices": self.artifact_channel_indices,
            "z_projection": self.z_projection,
        }


@dataclass(slots=True, frozen=True)
class FitsIOMeta:
    """
    Container for fits_io-specific metadata and any additional custom metadata. Represent the ground truth metadata for the image.
    """
    fits_io: ArtifactMeta = field(default_factory=ArtifactMeta)
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, payload_dict: Mapping[str, Any]) -> Self:
        """
        Create a FitsIOPayload instance from a dictionary representation.
        """
        fits_io_dict = payload_dict.get("fits_io", {})
        if not isinstance(fits_io_dict, Mapping):
            fits_io_dict = {}
    
        fits_io_meta = ArtifactMeta(**fits_io_dict)
        
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
                     artifact_type: str | None = None, 
                     created_by: str | None = None,
                     version: str | None = None,
                     derived_from: str | None = None,
                     axes: str | None = None, 
                     channel_labels: Sequence[str] | None = None, 
                     source_channel_indices: Sequence[int] | None = None, 
                     artifact_channel_indices: Sequence[int] | None = None, 
                     z_projection: Zproj = None, 
                     ) -> Self:
        """
        Return a new FitsIOPayload instance with updated fits_io metadata. If a parameter is None, the existing value is retained.
        """
        return replace(self,
            fits_io=replace(self.fits_io,
                            artifact_type=artifact_type if artifact_type is not None else self.fits_io.artifact_type,
                            created_by=created_by if created_by is not None else self.fits_io.created_by,
                            version=version if version is not None else self.fits_io.version,
                            derived_from=derived_from if derived_from is not None else self.fits_io.derived_from,
                            axes=axes if axes is not None else self.fits_io.axes,
                            channel_labels=list(channel_labels) if channel_labels is not None else self.fits_io.channel_labels,
                            source_channel_indices=list(source_channel_indices) if source_channel_indices is not None else self.fits_io.source_channel_indices,
                            artifact_channel_indices=list(artifact_channel_indices) if artifact_channel_indices is not None else self.fits_io.artifact_channel_indices,
                            z_projection=z_projection if z_projection is not None else self.fits_io.z_projection,)
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
    










