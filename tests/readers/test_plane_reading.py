"""Plane selection must decode pages instead of assembling TIFF series."""

import numpy as np
import pytest
import tifffile

from fits_io import FitsIO
from fits_io.readers.r_tiff import TiffReader


@pytest.mark.parametrize("axes,shape", [
    ("YX", (8, 9)), ("TYX", (4, 8, 9)),
    ("TCZYX", (3, 2, 4, 8, 9)), ("CTZYX", (2, 3, 4, 8, 9)),
    ("TZCYX", (3, 4, 2, 8, 9)),
])
@pytest.mark.parametrize("compression", [None, "zlib"])
def test_get_plane_reads_only_requested_page(tmp_path, monkeypatch, axes, shape, compression):
    path = tmp_path / "image.tif"
    array = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    tifffile.imwrite(path, array, photometric="minisblack", compression=compression,
                     metadata={"axes": axes})
    reader = FitsIO.from_path(path)
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray",
                        lambda *args, **kwargs: pytest.fail("Full-series read"))
    positions = {axis: shape[axes.index(axis)] - 1 if axis in axes else 0
                 for axis in "TCZ"}
    result = reader.get_plane(positions["T"],  positions["C"], positions["Z"])
    selection = tuple(positions.get(axis, slice(None)) for axis in axes)
    assert result.axes == "YX"
    np.testing.assert_array_equal(result.array, array[selection])
    with pytest.raises(IndexError):
        reader.get_plane(frame_index=shape[axes.index("T")] if "T" in axes else 1)


def test_plane_reading_uses_selected_series(tmp_path):
    path = tmp_path / "series.tif"
    first = np.zeros((3, 8, 9), dtype=np.uint16)
    second = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    with tifffile.TiffWriter(path) as writer:
        for array in (first, second):
            writer.write(array, photometric="minisblack", metadata={"axes": "TYX"})
    np.testing.assert_array_equal(TiffReader(path, series_idx=1).get_plane(2), second[2])


def test_imagej_hyperstack_plane_reading(tmp_path):
    path = tmp_path / "imagej.tif"
    array = np.arange(3 * 4 * 2 * 8 * 9, dtype=np.uint16).reshape(3, 4, 2, 8, 9)
    tifffile.imwrite(path, array, imagej=True, metadata={"axes": "TZCYX"})
    np.testing.assert_array_equal(FitsIO.from_path(path).get_plane(2, 1, 3).array,
                                  array[2, 3, 1])


def test_compressed_imagej_reuses_page_index(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "compressed.tif"
    array = np.arange(5 * 3 * 2 * 8 * 9, dtype=np.uint16).reshape(5, 3, 2, 8, 9)
    tifffile.imwrite(path, array, imagej=True, compression="zlib",
                     metadata={"axes": "TZCYX"})
    reader = TiffReader(path)
    assert len(reader._plane_offsets) == 30

    def fail_series(_self):
        pytest.fail("Rebuilt whole-stack metadata for a single plane")

    monkeypatch.setattr(tifffile.TiffFile, "series", property(fail_series))
    selections = [(0, 0, 0), (2, 1, 1), (4, 1, 2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        planes = list(pool.map(lambda selection: reader.get_plane(*selection), selections))
    for (frame, channel, z), plane in zip(selections, planes):
        np.testing.assert_array_equal(plane, array[frame, z, channel])
