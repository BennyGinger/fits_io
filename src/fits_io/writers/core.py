import errno
import tempfile
import time
from pathlib import Path
from typing import Any
import logging

from numpy.typing import NDArray
from tifffile import imwrite

from fits_io.metadata.tiff_meta import TiffWriteMeta


logger = logging.getLogger(__name__)


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Publish a completed TIFF, allowing temporary sharing locks to clear."""
    delays = (0.25, 0.5, 1.0, 2.0, 4.0)
    for attempt in range(len(delays) + 1):
        try:
            source.replace(destination)
            return
        except OSError as error:
            busy = error.errno == errno.EBUSY or getattr(error, "winerror", None) in {32, 33}
            if not busy or attempt == len(delays):
                raise
            logger.warning(
                "TIFF rename is busy for %s; retrying in %.2f seconds (%d/%d).",
                destination, delays[attempt], attempt + 1, len(delays),
            )
            time.sleep(delays[attempt])


def save_tiff(img_array: NDArray, 
              save_path: Path, 
              metadata: TiffWriteMeta, 
              compression: str | None = 'zlib',
              *, compressionargs: dict[str, Any] | None = None
              ) -> None:
    """
    Save a NumPy array to a TIFF file with the specified metadata and compression.
    """
    predictor = 2 if compression in {"zlib", "deflate", "lzma"} and img_array.dtype.kind in "iu" else None
    logger.debug(f"compression={compression} predictor={predictor} dtype={img_array.dtype} shape={img_array.shape} size={img_array.size}")
    
    if img_array.size == 0:
        raise ValueError("Cannot save empty array to TIFF. The input array has zero elements.")
    
    # Use a temporary file to ensure atomic write
    with tempfile.NamedTemporaryFile(dir=save_path.parent, suffix=save_path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        imwrite(tmp_path,
                img_array,
                imagej=True,
                metadata=metadata.imagej_meta,
                resolution=metadata.resolution,
                predictor=predictor,
                extratags=metadata.extratags,
                compression=compression, compressionargs=compressionargs,)
        
        _replace_with_retry(tmp_path, save_path)
        logger.debug(f"Saved TIFF file at {save_path}")
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary TIFF %s", tmp_path, exc_info=True)
        logger.exception(f"Failed to save TIFF file at {save_path}")
        raise
