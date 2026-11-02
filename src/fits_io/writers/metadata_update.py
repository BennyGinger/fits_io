"""Atomic private-metadata replacement without decoding or changing image pixels."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import tifffile
from pathlib import Path
from fits_io.metadata.codec import FITS_TAG
from fits_io.metadata.models import FitsIOMeta
from fits_io.writers.core import _replace_with_retry


def rewrite_metadata(path: Path, metadata: FitsIOMeta, *, output_path: Path | None = None) -> Path:
    """
    Replace the private TIFF metadata atomically, preserving encoded pixel data.
    """
    target = output_path or path
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=target.suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(path, temporary)
        with tifffile.TiffFile(temporary, mode="r+") as tif:
            tag = tif.pages.first.tags.get(FITS_TAG)
            if tag is None:
                raise ValueError("This TIFF has no private metadata tag to update.")
            tag.overwrite(json.dumps(metadata.dump(), ensure_ascii=False).encode())
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
