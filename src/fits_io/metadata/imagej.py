from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fits_io.metadata.codec import encode_metadata
from fits_io.metadata.models import ChannelMeta, InfoSummary, ResolutionMeta, StackMeta
from fits_io.metadata.models import TiffMetadata
from fits_io.readers._types import StatusFlag

if TYPE_CHECKING:
    from fits_io.metadata.context import MetadataBuildContext


def build_tiff_metadata(ctx: MetadataBuildContext, payload: dict[str, Any]) -> TiffMetadata:
    """Build final TIFF metadata from resolved context and final private payload."""
    channel_meta, resolution_meta, stack_meta, info = _build_models(n_channels=ctx.n_channels,
                                                                   labels=ctx.labels,
                                                                   axes=ctx.axes,
                                                                   interval=ctx.interval,
                                                                   resolution=ctx.resolution, 
                                                                   status=ctx.status, 
                                                                   user_name=ctx.user_name, 
                                                                   payload=payload)
    
    metadata_dict = _assemble_imagej_metadata(channel_meta, resolution_meta, stack_meta, info)
    extratags = encode_metadata(payload)
    return TiffMetadata(imagej_meta=metadata_dict, resolution=resolution_meta.resolution, extratags=extratags)

def _build_models(*, n_channels: int, labels: list[str] | None, axes: str, interval: float | None, resolution: tuple[float, float] | None, status: StatusFlag, user_name: str, payload: dict[str, Any]) -> tuple[ChannelMeta, ResolutionMeta, StackMeta, InfoSummary]:
    """
    Build ImageJ-facing metadata model objects.
    """
    channel_meta = ChannelMeta(channel_number=n_channels, labels=labels)
    resolution_meta = ResolutionMeta(resolution)
    stack_meta = StackMeta(axes=axes, finterval=interval)
    info = InfoSummary(status=status, user_name=user_name, chosen_labels=labels, current_meta=payload)
    return channel_meta, resolution_meta, stack_meta, info

def _assemble_imagej_metadata(channel_meta: ChannelMeta, resolution_meta: ResolutionMeta, stack_meta: StackMeta, info: InfoSummary) -> dict[str, Any]:
    """
    Assemble final ImageJ metadata dict from viewer-facing components.
    """
    metadata_dict = stack_meta.to_dict()
    metadata_dict['Info'] = info.render()
    metadata_dict.update(channel_meta.to_dict())
    metadata_dict.update(resolution_meta.to_dict())
    return metadata_dict


