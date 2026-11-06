"""ND2 frame addressing and memory-map lifetime regression checks."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import nd2
import numpy as np
from numpy.typing import NDArray
import pytest

from fits_io.readers.r_nd2 import Nd2Reader


@pytest.mark.parametrize("loop_axes", ["TPZ", "ZPT", "PTZ"])
def test_nd2_plane_reads_only_selected_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             loop_axes: str,
                                             ) -> None:
    """
    Address loops in file order, select C, and copy pixels while their mapping is alive.
    """
    dimensions = {"T": 3, "P": 2, "Z": 4}
    sizes = {axis: dimensions[axis] for axis in loop_axes} | {"C": 2, "Y": 3, "X": 5}
    reads: list[int] = []

    class FakeND2:
        def __init__(self, path: Path) -> None:
            """
            Supply metadata and a temporary frame buffer.
            """
            self.sizes = sizes
            self.shape = tuple(sizes.values())
            self.metadata = SimpleNamespace(channels=None)
            self.experiment = []
            self.dtype = np.dtype("uint16")
            self.buffer: NDArray[np.uint16] = np.empty((2, 3, 5), dtype=np.uint16)

        def __enter__(self) -> Self:
            """
            Open the simulated mapping.
            """
            return self

        def __exit__(self, *args: Any) -> None:
            """
            Invalidate pixels when closing, like an ND2 memory map.
            """
            self.buffer.fill(0)

        def read_frame(self, index: int) -> NDArray[np.uint16]:
            """
            Return channels with distinguishable sequence and channel values.
            """
            reads.append(index)
            self.buffer[0].fill(index * 2)
            self.buffer[1].fill(index * 2 + 1)
            return self.buffer

    monkeypatch.setattr(nd2, "ND2File", FakeND2)
    reader = Nd2Reader(tmp_path / "input.nd2", series_idx=1)
    assert reader.dtype == np.dtype("uint16")
    coordinates = {"T": 2, "P": 1, "Z": 3}
    expected = int(np.ravel_multi_index(tuple(coordinates[axis] for axis in loop_axes),
                                        tuple(dimensions[axis] for axis in loop_axes)))
    plane = reader.get_plane(2, 1, 3)
    assert reads == [expected]
    assert plane.shape == (3, 5)
    np.testing.assert_array_equal(plane, np.full((3, 5), expected * 2 + 1, dtype=np.uint16))
    for indices in [(-1, 0, 0), (3, 0, 0), (0, 2, 0), (0, 0, 4)]:
        with pytest.raises(IndexError):
            reader.get_plane(*indices)
    assert reads == [expected]


def test_nd2_plane_without_navigation_axes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           fake_nd2_file_noP: Any,
                                           ) -> None:
    """
    Support single channels and reject unavailable navigation before pixel access.
    """
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_noP)
    reader = Nd2Reader(tmp_path / "input.nd2")
    monkeypatch.setattr(reader, "_sizes", {"Y": 3, "X": 4})
    monkeypatch.setattr(reader, "_axes", "YX")
    monkeypatch.setattr(reader, "_shape", (3, 4))
    file_type = type(fake_nd2_file_noP(tmp_path / "input.nd2"))
    monkeypatch.setattr(file_type, "read_frame", lambda self, index: np.ones((3, 4)), raising=False)
    assert reader.get_plane().shape == (3, 4)
    with pytest.raises(IndexError):
        reader.get_plane(frame_index=1)
