# tests/conftest.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from fits_io.readers.factory import ArrAxis, ImageReader, StatusFlag, Zproj


# ============================================================
# Generic lightweight helpers used across multiple test modules
# ============================================================

@pytest.fixture
def fake_tiff_metadata():
    """Factory returning a simple stand-in for fits_io.metadata.TiffMetadata."""
    def _make(*, imagej_meta: dict[str, Any] | None = None,
              resolution: tuple[float, float] | None = None,
              extratags: list[tuple[Any, ...]] | None = None):
        return SimpleNamespace(
            imagej_meta=imagej_meta or {},
            resolution=resolution,
            extratags=extratags or [],
        )
    return _make


@pytest.fixture
def arr_empty() -> NDArray:
    return np.array([], dtype=np.uint8)


@pytest.fixture
def arr_u8_2d() -> NDArray:
    return np.ones((4, 4), dtype=np.uint8)


@pytest.fixture
def arr_u16_2d() -> NDArray:
    return np.ones((5, 6), dtype=np.uint16)


# ==========================
# DummyReader (kept as-is)
# ==========================
# (this is the one you already have; included here for completeness)
@dataclass
class DummyReader(ImageReader):
    _series_number: int = 1
    _labels: list[str] | None = None

    array: NDArray | list[NDArray] | None = None
    channel_array: NDArray | list[NDArray] | None = None

    called_get_array: int = 0
    called_get_channel: int = 0
    last_get_channel_arg: object = None

    def __post_init__(self) -> None:
        if self._labels is None:
            self._labels = ["C_1", "C_2", "C_3"]
        if self.array is None:
            self.array = np.ones((2, 3), dtype=np.uint8)
        if self.channel_array is None:
            self.channel_array = np.ones((2, 3), dtype=np.uint8)

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return True

    @property
    def axes(self) -> list[str]:
        return ["YX"]

    @property
    def compression_method(self) -> str | None:
        return None

    @property
    def status(self) -> StatusFlag:
        return "active"

    @property
    def export_status(self) -> str:
        return ""

    @property
    def channel_number(self) -> list[int]:
        return [len(self._labels or [])]

    def _native_channel_labels(self) -> list[str] | None:
        return self._labels

    @property
    def series_number(self) -> int:
        return self._series_number

    def axis_index(self, axis: ArrAxis) -> list[int | None]:
        return [None] * self.series_number

    @property
    def resolution(self) -> list[tuple[float, float] | None]:
        return [None] * self.series_number

    @property
    def interval(self) -> float | None:
        return None

    @property
    def custom_metadata(self) -> Mapping[str, Any]:
        return {}

    def get_array(self, z_projection: Zproj = None) -> NDArray | list[NDArray]:
        self.called_get_array += 1
        return self.array  # type: ignore[return-value]

    def get_channel(
        self,
        channel: int | str | Sequence[int | str],
        z_projection: Zproj = None,
    ) -> NDArray | list[NDArray]:
        self.called_get_channel += 1
        self.last_get_channel_arg = channel
        return self.channel_array  # type: ignore[return-value]


@pytest.fixture
def dummy_reader(tmp_path: Path) -> DummyReader:
    return DummyReader(img_path=tmp_path / "img.tif")


# ============================================================
# TIFF fakes (moved from test_image_reader.py)
# ============================================================

class FakeTiffTags:
    """Acts like tifffile.TiffTags for what we use: iter + get(name_or_code)."""
    def __init__(self, tags_list):
        self._tags = list(tags_list)

    def __iter__(self):
        return iter(self._tags)

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, key, default=None):
        for t in self._tags:
            if isinstance(key, int) and getattr(t, "code", None) == key:
                return t
            if isinstance(key, str) and getattr(t, "name", None) == key:
                return t
        return default


class FakeTag:
    def __init__(self, name, value, code=None):
        self.name = name
        self.value = value
        self.code = code


class FakePage:
    def __init__(self, tags):
        self.tags = FakeTiffTags(tags)


class FakeSeries:
    def __init__(self, axes, shape, pages=None):
        self.axes = axes
        self.shape = shape
        self.pages = pages or []

    def asarray(self):
        return np.zeros(self.shape, dtype=np.uint8)


class FakeTiff:
    def __init__(self, axes="YX", shape=(5, 6), tags=None, imagej_metadata=None, series=None):
        page = FakePage(tags or [])
        if series is None:
            self.series = [FakeSeries(axes, shape, pages=[page])]
        else:
            self.series = series
        self.pages = [page]
        self.imagej_metadata = imagej_metadata

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_tiff_file_full() -> Callable[[Path], FakeTiff]:
    def _factory(path: Path):
        tags = [
            FakeTag("Compression", 1),
            FakeTag("XResolution", (2, 1)),  # xres=2 -> 0.5 um/px
            FakeTag("YResolution", (4, 1)),  # yres=4 -> 0.25 um/px
        ]
        ij = {"finterval": 11.0}
        page = FakePage(tags)
        series = [
            FakeSeries("CYX", (3, 5, 6), pages=[page]),
            FakeSeries("CYX", (3, 5, 6), pages=[page]),
        ]
        return FakeTiff(series=series, tags=tags, imagej_metadata=ij)
    return _factory


@pytest.fixture
def fake_tiff_file_no_series() -> Callable[[Path], FakeTiff]:
    def _factory(path: Path):
        tags = [FakeTag("Compression", 1)]
        return FakeTiff(axes="YX", shape=(5, 6), tags=tags, imagej_metadata={})
    return _factory


@pytest.fixture
def fake_tiff_file_no_series_axes() -> Callable[[Path], FakeTiff]:
    def _factory(path: Path):
        tags = [FakeTag("Compression", 1)]
        return FakeTiff(axes="ZYX", shape=(2, 5, 6), tags=tags, imagej_metadata={})
    return _factory


@pytest.fixture
def fake_tiff_file_series_S0() -> Callable[[Path], FakeTiff]:
    def _factory(path: Path):
        tags = [FakeTag("Compression", 1)]
        page = FakePage(tags)
        series = [
            FakeSeries("YX", (5, 6), pages=[page]),
            FakeSeries("YX", (5, 6), pages=[page]),
        ]
        return FakeTiff(series=series, tags=tags, imagej_metadata={})
    return _factory


# ============================================================
# ND2 fakes (moved from test_image_reader.py)
# ============================================================

class _LoopParams:
    def __init__(self, periodMs=None, periods=None):
        self.periodMs = periodMs
        self.periods = periods

class _Period:
    def __init__(self, periodMs):
        self.periodMs = periodMs

class _ExpLoop:
    def __init__(self, type_, parameters):
        self.type = type_
        self.parameters = parameters

class _Channel:
    def __init__(self, volume=None):
        self.volume = volume

class _Volume:
    def __init__(self, axesCalibration=None):
        self.axesCalibration = axesCalibration


class FakeND2:
    def __init__(self, sizes, metadata, experiment):
        self.sizes = sizes
        self.metadata = metadata
        self.experiment = experiment

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _Meta_no_channels():
    class M: ...
    return M()

def _Meta_channels_no_volume():
    class M:
        channels = [_Channel(volume=None)]
    return M()

def _Meta_channels_no_calib():
    class M:
        channels = [_Channel(volume=_Volume(axesCalibration=None))]
    return M()

def _Meta_channels_calib():
    class M:
        channels = [_Channel(volume=_Volume(axesCalibration=(0.3223335, 0.3223335)))]
    return M()


@pytest.fixture
def nd2_meta_factories():
    """So test_image_reader can parametrize meta factories without defining them inline."""
    return {
        "no_channels": _Meta_no_channels,
        "no_volume": _Meta_channels_no_volume,
        "no_calib": _Meta_channels_no_calib,
        "calib": _Meta_channels_calib,
    }


@pytest.fixture
def fake_nd2_file_basic():
    def _factory(path: Path):
        sizes = {"T": 2, "Z": 3, "C": 2, "P": 2, "Y": 5, "X": 6}  # TZCPYX
        meta = _Meta_channels_calib()
        exploop = [_ExpLoop("TimeLoop", _LoopParams(periodMs=5000))]
        return FakeND2(sizes, meta, exploop)
    return _factory


@pytest.fixture
def fake_nd2_file_meta():
    def _wrap(meta_factory):
        def _factory(path: Path):
            sizes = {"T": 1, "C": 1, "Y": 5, "X": 6}
            meta = meta_factory()
            exploop = []
            return FakeND2(sizes, meta, exploop)
        return _factory
    return _wrap


@pytest.fixture
def fake_nd2_file_T1():
    def _factory(path: Path):
        sizes = {"T": 1, "Y": 5, "X": 6}
        meta = _Meta_no_channels()
        exploop = []
        return FakeND2(sizes, meta, exploop)
    return _factory


@pytest.fixture
def fake_nd2_file_timeloop():
    def _factory(path: Path):
        sizes = {"T": 3, "Y": 5, "X": 6}
        meta = _Meta_no_channels()
        exploop = [_ExpLoop("TimeLoop", _LoopParams(periodMs=5000))]
        return FakeND2(sizes, meta, exploop)
    return _factory


@pytest.fixture
def fake_nd2_file_netimeloop():
    def _factory(path: Path):
        sizes = {"T": 3, "Y": 5, "X": 6}
        meta = _Meta_no_channels()
        exploop = [_ExpLoop("NETimeLoop", _LoopParams(periods=[_Period(periodMs=7000)]))]
        return FakeND2(sizes, meta, exploop)
    return _factory


@pytest.fixture
def fake_nd2_file_noP():
    def _factory(path: Path):
        sizes = {"T": 2, "Y": 5, "X": 6}
        meta = _Meta_channels_calib()
        exploop = []
        return FakeND2(sizes, meta, exploop)
    return _factory


# ============================================================
# Writer harness (as before) + a "tiff" variant
# ============================================================

@dataclass
class WriterHarness:
    tmp_path: Path
    writer_mod: Any

    save_dirs: list[Path] = field(default_factory=list)
    already_converted: bool = False

    used_channels: list[str] = field(default_factory=lambda: ["C_1"])
    export_all_flag: bool = True
    arrays: list[NDArray] = field(default_factory=lambda: [np.ones((2, 2), dtype=np.uint8)])

    saved: list[dict[str, Any]] = field(default_factory=list)
    md_calls: list[dict[str, Any]] = field(default_factory=list)

    def make_meta(self, **kwargs: Any) -> Any:
        return SimpleNamespace(imagej_meta={"kwargs": kwargs}, resolution=None, extratags=[])


@pytest.fixture
def writer_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WriterHarness:
    import fits_io.writers.api as writer_mod

    h = WriterHarness(tmp_path=tmp_path, writer_mod=writer_mod)
    h.save_dirs = [tmp_path / "img_s1"]

    monkeypatch.setattr(writer_mod, "get_save_dirs", lambda _r: h.save_dirs)
    monkeypatch.setattr(writer_mod, "mkdirs_paths", lambda dirs: dirs)
    monkeypatch.setattr(writer_mod, "build_output_path", lambda d, *, save_name: d / save_name)
    monkeypatch.setattr(writer_mod, "image_converted", lambda _dirs: h.already_converted)

    monkeypatch.setattr(
        writer_mod,
        "resolve_channel_labels",
        lambda _labels, _n, _export: (h.used_channels, h.export_all_flag),
    )
    monkeypatch.setattr(
        writer_mod,
        "get_array_to_export",
        lambda _r, _used, _flag, _zp: h.arrays,
    )

    def fake_build_imagej_metadata(*args: Any, **kwargs: Any) -> Any:
        h.md_calls.append(kwargs)
        return h.make_meta(**kwargs)

    monkeypatch.setattr(writer_mod, "build_imagej_metadata", fake_build_imagej_metadata)

    def fake_save_tiff(array: NDArray, path: Path, meta: Any, *, compression: str | None = "zlib") -> None:
        h.saved.append({"array": array, "path": path, "meta": meta, "compression": compression})

    monkeypatch.setattr(writer_mod, "_save_tiff", fake_save_tiff)
    return h


@pytest.fixture
def writer_harness_tiff(monkeypatch: pytest.MonkeyPatch, writer_harness: WriterHarness, dummy_reader: DummyReader) -> WriterHarness:
    """
    Same harness, but makes DummyReader pass the `isinstance(..., TiffReader)` guard
    used in set_status/set_channel_labels. :contentReference[oaicite:2]{index=2}
    """
    import fits_io.writers.api as writer_mod
    monkeypatch.setattr(writer_mod, "TiffReader", type(dummy_reader))
    return writer_harness
