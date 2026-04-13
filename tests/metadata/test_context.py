from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from fits_io.metadata.context import MetadataBuildContext, resolve_build_context
from fits_io.readers.protocol import ImageReader


@dataclass
class ReaderStub:
    axes: list[str] = field(default_factory=lambda: ['TZCYX'])
    interval: float | None = 11.0
    channel_number: list[int] = field(default_factory=lambda: [2])
    resolution: list[tuple[float, float] | None] = field(default_factory=lambda: [(0.5, 0.25)])
    custom_metadata: Mapping[str, Any] = field(default_factory=dict)
    channel_labels: list[str] = field(default_factory=lambda: ['C_1', 'C_2'])


def test_resolve_build_context_basic_fields():
    reader = ReaderStub(custom_metadata={'legacy': 'kept'})
    ctx = resolve_build_context(cast(ImageReader, reader), channel_labels=['GFP', 'RFP'])
    assert isinstance(ctx, MetadataBuildContext)
    assert ctx.n_channels == 2
    assert ctx.labels == ['GFP', 'RFP']
    assert ctx.axes == 'TZCYX'
    assert ctx.base_payload['legacy'] == 'kept'
    assert ctx.interval == 11.0
    assert ctx.resolution == (0.5, 0.25)


def test_resolve_build_context_drops_z_when_projected():
    reader = ReaderStub(axes=['TZCYX'])
    ctx = resolve_build_context(cast(ImageReader, reader), z_projection='max')
    assert ctx.axes == 'TCYX'


def test_resolve_build_context_drops_c_for_single_channel():
    reader = ReaderStub(axes=['TCYX'], channel_number=[3], channel_labels=['GFP'])
    ctx = resolve_build_context(cast(ImageReader, reader), channel_labels='GFP')
    assert ctx.n_channels == 1
    assert ctx.labels == ['GFP']
    assert ctx.axes == 'TYX'


def test_resolve_build_context_uses_series_index():
    reader = ReaderStub(axes=['CYX', 'TCYX'], channel_number=[2, 3], resolution=[(0.5, 0.5), (0.3, 0.3)], channel_labels=['C_1', 'C_2', 'C_3'])
    ctx = resolve_build_context(cast(ImageReader, reader), series_index=1)
    assert ctx.axes == 'TCYX'
    assert ctx.n_channels == 3
    assert ctx.resolution == (0.3, 0.3)


def test_resolve_build_context_passes_through_source_channel_identity():
    reader = ReaderStub(channel_number=[3], channel_labels=['C_1', 'C_2', 'C_3'])
    ctx = resolve_build_context(cast(ImageReader, reader), source_channel_indices=[1, 2], source_channel_count=3)
    assert ctx.source_channel_indices == [1, 2]
    assert ctx.source_channel_count == 3