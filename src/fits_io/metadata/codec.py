import json
from typing import Any, Mapping

from fits_io.metadata.provenance import FITS_TAG
from fits_io.readers._types import ExtraTags


def encode_metadata(payload: Mapping[str, Any]) -> ExtraTags | None:
    """Encode private metadata payload for storage in TIFF extra tags."""
    if payload:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return [(FITS_TAG, "B", len(raw), raw, True)]
    return None


def decode_metadata(raw: bytes | bytearray | str) -> dict[str, Any]:
    """Decode metadata payload from TIFF tag raw bytes/string."""
    if isinstance(raw, (bytes, bytearray)):
        return json.loads(raw.decode("utf-8", "replace"))
    return json.loads(raw)