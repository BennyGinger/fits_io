from pathlib import Path

import numpy as np

from fits_io.writers import core


def test_fast_lossless_compression_round_trip(tmp_path: Path) -> None:
    from fits_io.metadata.tiff_meta import TiffWriteMeta
    from tifffile import TiffFile, imread
    array = np.zeros((4, 30, 31), dtype=np.uint16)
    array[:, 4:20, 5:21] = 4
    array[2, 8:10, 9:11] = 2
    path = tmp_path / 'fast-mask.tif'
    metadata = TiffWriteMeta(imagej_meta={'axes': 'TYX'})
    core.save_tiff(array, path, metadata, compression='zlib', compressionargs={'level': 1})
    np.testing.assert_array_equal(imread(path), array)
    with TiffFile(path) as tiff:
        assert tiff.series[0].axes == 'TYX'
        assert tiff.series[0].dtype == np.uint16
        assert int(tiff.pages[0].compression) in (8, 32946)
