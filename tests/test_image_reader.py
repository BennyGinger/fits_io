import numpy as np
import pytest
from pathlib import Path

import fits_io.image_reader as m


# --------------------------
# get_reader
# --------------------------

def test_get_reader_missing_file_raises(tmp_path: Path):
    p = tmp_path / "missing.tif"
    with pytest.raises(m.FileNotFoundError):
        m.get_reader(p)


def test_get_reader_unsupported_suffix_raises(tmp_path: Path):
    p = tmp_path / "x.bmp"
    p.write_bytes(b"not really an image")
    with pytest.raises(m.UnsupportedFileTypeError) as e:
        m.get_reader(p)
    msg = str(e.value)
    assert "Unsupported file type" in msg
    assert ".tif" in msg and ".nd2" in msg  # mentions supported


def test_get_reader_tif(tmp_path: Path, monkeypatch):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    # avoid real tifffile access in __post_init__
    monkeypatch.setattr(m, "TiffFile", _FakeTiffFile_no_series)

    r = m.get_reader(p)
    assert isinstance(r, m.TiffReader)


def test_get_reader_nd2(tmp_path: Path, monkeypatch):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_basic)

    r = m.get_reader(p)
    assert isinstance(r, m.Nd2Reader)
    

# --------------------------
# TiffReader behavior
# --------------------------

def test_tiff_can_read():
    assert m.TiffReader.can_read(Path("a.tif"))
    assert m.TiffReader.can_read(Path("a.TIFF"))
    assert not m.TiffReader.can_read(Path("a.nd2"))


def test_tiff_axes_channel_series_resolution_interval(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    # Fake a TIFF with axes "SCYX", C=3, S=2, and resolution tags
    monkeypatch.setattr(m, "TiffFile", _FakeTiffFile_full)

    r = m.TiffReader(p)
    assert r.axes == "SCYX"
    assert r.channel_number == 3
    assert r.serie_axis_index == 0  # "S" first in "SCYX"
    assert r.resolution == (0.5, 0.25)  # from fake tags
    assert r.interval == 11.0  # finterval from fake imagej_metadata


def test_tiff_get_array_no_series(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m, "TiffFile", _FakeTiffFile_no_series_axes)
    monkeypatch.setattr(m, "imread", lambda _: np.zeros((2, 3, 4), dtype=np.uint8))

    r = m.TiffReader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3, 4)


def test_tiff_get_array_with_series_splits(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m, "TiffFile", _FakeTiffFile_series_S0)
    monkeypatch.setattr(m, "imread", lambda _: np.zeros((2, 5, 6), dtype=np.uint8))  # S,Y,X

    r = m.TiffReader(p)
    out = r.get_array()
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (5, 6)  # squeezed S axis


# --------------------------
# Nd2Reader behavior
# --------------------------

def test_nd2_can_read():
    assert m.Nd2Reader.can_read(Path("a.nd2"))
    assert m.Nd2Reader.can_read(Path("a.ND2"))
    assert not m.Nd2Reader.can_read(Path("a.tif"))


def test_nd2_axes_channel_series(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_basic)

    r = m.Nd2Reader(p)
    assert r.axes == "TZCPYX"  # from fake sizes keys
    assert r.channel_number == 2
    assert r.serie_axis_index == r.axes.index("P")


# Meta factory functions for parametrize
def _Meta_no_channels():
    class M:
        ...
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


@pytest.mark.parametrize(
    "meta_factory, expected",
    [
        (_Meta_no_channels, (1.0, 1.0)),
        (_Meta_channels_no_volume, (1.0, 1.0)),
        (_Meta_channels_no_calib, (1.0, 1.0)),
        (_Meta_channels_calib, (0.3223, 0.3223)),
    ],
)
def test_nd2_resolution_defensive(monkeypatch, tmp_path: Path, meta_factory, expected):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")

    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_meta(meta_factory))

    r = m.Nd2Reader(p)
    assert r.resolution == expected


def test_nd2_interval_none_when_no_time(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_T1)

    r = m.Nd2Reader(p)
    assert r.interval is None


def test_nd2_interval_timeloop(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_timeloop)

    r = m.Nd2Reader(p)
    assert r.interval == 5  # 5000ms -> 5s


def test_nd2_interval_netimeloop(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_netimeloop)

    r = m.Nd2Reader(p)
    assert r.interval == 7  # 7000ms -> 7s


def test_nd2_get_array_no_series(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_noP)
    monkeypatch.setattr(m.nd2, "imread", lambda _: np.zeros((3, 4), dtype=np.uint16))

    r = m.Nd2Reader(p)
    out = r.get_array()
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)


def test_nd2_get_array_with_series_splits(monkeypatch, tmp_path: Path):
    p = tmp_path / "x.nd2"
    p.write_bytes(b"fake")
    monkeypatch.setattr(m.nd2, "ND2File", _FakeND2File_basic)

    # axes "TZCPYX": P is index 3 in our fake
    # Make shape where P=2 to split into 2 series
    monkeypatch.setattr(m.nd2, "imread", lambda _: np.zeros((2, 3, 2, 2, 5, 6), dtype=np.uint16))

    r = m.Nd2Reader(p)
    out = r.get_array()
    assert isinstance(out, list)
    assert len(out) == 2
    # P axis removed by squeeze
    assert out[0].shape == (2, 3, 2, 5, 6)

# ==========================
# Fakes / stubs
# ==========================
class _FakeTiffTags:
    """Acts like tifffile.TiffTags for what we use: iter + get(tag_code)."""
    def __init__(self, tags_list):
        self._tags = list(tags_list)

    def __iter__(self):
        return iter(self._tags)

    def get(self, code, default=None):
        # tifffile lets you lookup by numeric tag code.
        for t in self._tags:
            if getattr(t, "code", None) == code:
                return t
        return default


class _FakeTag:
    def __init__(self, name, value, code=None):
        self.name = name
        self.value = value
        self.code = code  # <-- needed for PIPELINE_TAG lookup


class _FakePage:
    def __init__(self, tags):
        self.tags = _FakeTiffTags(tags)


class _FakeSeries:
    def __init__(self, axes, shape):
        self.axes = axes
        self.shape = shape


class _FakeTiff:
    def __init__(self, axes="YX", shape=(5, 6), tags=None, imagej_metadata=None):
        self.series = [_FakeSeries(axes, shape)]
        self.pages = [_FakePage(tags or [])]
        self.imagej_metadata = imagej_metadata

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _FakeTiffFile_full(path):
    # axes "SCYX": S=2, C=3, Y=5, X=6
    tags = [
        _FakeTag("XResolution", (2, 1)),  # xres=2 -> um/px=0.5
        _FakeTag("YResolution", (4, 1)),  # yres=4 -> um/px=0.25
    ]
    ij = {"finterval": 11.0}
    return _FakeTiff(axes="SCYX", shape=(2, 3, 5, 6), tags=tags, imagej_metadata=ij)


def _FakeTiffFile_no_series(path):
    # minimal safe for __post_init__
    return _FakeTiff(axes="YX", shape=(5, 6), tags=[], imagej_metadata={})


def _FakeTiffFile_no_series_axes(path):
    return _FakeTiff(axes="ZYX", shape=(2, 5, 6), tags=[], imagej_metadata={})


def _FakeTiffFile_series_S0(path):
    # series axis "S" first
    return _FakeTiff(axes="SYX", shape=(2, 5, 6), tags=[], imagej_metadata={})


# ---- ND2 fakes ----

class _FakeND2:
    def __init__(self, sizes, metadata, experiment):
        self.sizes = sizes
        self.metadata = metadata
        self.experiment = experiment

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


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


def _FakeND2File_basic(path):
    sizes = {"T": 2, "Z": 3, "C": 2, "P": 2, "Y": 5, "X": 6}  # axes TZCPYX
    meta = _Meta_channels_calib()
    exploop = [_ExpLoop("TimeLoop", _LoopParams(periodMs=5000))]
    return _FakeND2(sizes, meta, exploop)


def _FakeND2File_meta(meta_factory):
    def _factory(path):
        sizes = {"T": 1, "C": 1, "Y": 5, "X": 6}
        meta = meta_factory()
        exploop = []
        return _FakeND2(sizes, meta, exploop)
    return _factory


def _FakeND2File_T1(path):
    sizes = {"T": 1, "Y": 5, "X": 6}
    meta = _Meta_no_channels()
    exploop = []
    return _FakeND2(sizes, meta, exploop)


def _FakeND2File_timeloop(path):
    sizes = {"T": 3, "Y": 5, "X": 6}
    meta = _Meta_no_channels()
    exploop = [_ExpLoop("TimeLoop", _LoopParams(periodMs=5000))]
    return _FakeND2(sizes, meta, exploop)


def _FakeND2File_netimeloop(path):
    sizes = {"T": 3, "Y": 5, "X": 6}
    meta = _Meta_no_channels()
    exploop = [_ExpLoop("NETimeLoop", _LoopParams(periods=[_Period(periodMs=7000)]))]
    return _FakeND2(sizes, meta, exploop)


def _FakeND2File_noP(path):
    sizes = {"T": 2, "Y": 5, "X": 6}  # no P axis
    meta = _Meta_channels_calib()
    exploop = []
    return _FakeND2(sizes, meta, exploop)