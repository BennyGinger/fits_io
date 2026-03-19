import tempfile
from pathlib import Path
import logging

from numpy.typing import NDArray
from tifffile import imwrite

from fits_io.metadata.models import TiffMetadata


logger = logging.getLogger(__name__)

def save_tiff(img_array: NDArray, save_path: Path, metadata: TiffMetadata, compression: str | None = 'zlib') -> None:
    
    predictor = 2 if compression in {"zlib", "deflate", "lzma"} else None
    logger.debug(f"compression={compression} predictor={predictor} dtype={img_array.dtype} shape={img_array.shape} size={img_array.size}")
    
    if img_array.size == 0:
        raise ValueError("Cannot save empty array to TIFF. The input array has zero elements.")
    
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
                compression=compression,)
        
        tmp_path.replace(save_path)
        logger.debug(f"Saved TIFF file at {save_path}")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        logger.exception(f"Failed to save TIFF file at {save_path}")
        raise