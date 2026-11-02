"""Generic input revision references; no workflow or filename conventions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from fits_io.metadata.models import FitsIOMeta


def input_reference(path: Path, metadata: FitsIOMeta, output_directory: Path) -> dict[str, Any]:
    """
    Record the actual input revision using a portable relative path.
    """
    return {"artifact_path": Path(os.path.relpath(path.resolve(), output_directory.resolve())).as_posix(),
            "artifact_type": metadata.fits_io.artifact_type,
            "created_at": metadata.fits_io.created_at,
            "external": metadata.fits_io.artifact_type is None}
