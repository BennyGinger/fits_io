from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import json

import numpy as np
from numpy.typing import NDArray

COLOR_MAP = {
        "red":     (1, 0, 0),
        "green":   (0, 1, 0),
        "blue":    (0, 0, 1),
        "cyan":    (0, 1, 1),
        "magenta": (1, 0, 1),
        "yellow":  (1, 1, 0),
        "gray":    (1, 1, 1),
    }

LABEL_TO_COLOR = {
    "blue": 'blue',
    "bfp": 'blue',
    "dapi": 'blue',
    "cyan": 'cyan',
    "cfp": 'cyan',
    "yellow": 'yellow',
    "cy3": 'yellow',
    "green": 'green',
    "gcamp": 'green',
    "gfp": 'green',
    "egfp": 'green',
    "fitc": 'green',
    "magenta": 'magenta',
    "mch": 'magenta',
    "ired": 'magenta',
    "irfp": 'magenta',
    "red": 'red',
    "pinky": 'red',
    "mkate2": 'red',
    "scarlet": 'red',
    "geco": 'red',
    "mcherry": 'red',
    "tritc": 'red',
    "rfp": 'red',
    'gray': 'gray',
    'grey': 'gray',
    'bf': 'gray',
    'brightfield': 'gray',
}


@dataclass(slots=True, frozen=True)
class InfoSummary:
    payload: Mapping[str, Any] | None

    @staticmethod
    def _toml_value(value: Any) -> str:
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, tuple):
            return json.dumps(list(value), ensure_ascii=False)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(str(value), ensure_ascii=False)

    @classmethod
    def _append_toml_sections(cls, lines: list[str], section: str, value: Mapping[str, Any]) -> None:
        lines.append(f"[{section}]")

        for key, child in value.items():
            if isinstance(child, Mapping):
                continue
            lines.append(f"{key} = {cls._toml_value(child)}")

        lines.append("")

        for key, child in value.items():
            if isinstance(child, Mapping):
                cls._append_toml_sections(lines, f"{section}.{key}", child)

    def render(self) -> str:
        delimiter = "----------------------"
        lines = [
            delimiter,
            "ARTIFACT METADATA",
            delimiter,
            "",
        ]

        if not self.payload:
            return "\n".join(lines) + "\n"

        for i, (key, value) in enumerate(self.payload.items()):
            if i > 0:
                lines.append("---")

            if isinstance(value, Mapping):
                self._append_toml_sections(lines, key, value)
            else:
                lines.append(f"{key} = {self._toml_value(value)}")
                lines.append("")

        return "\n".join(lines) + "\n"
   
            
def make_color_lut(color: str) -> NDArray[np.uint8]:
    """Return ImageJ-style LUT: shape (3, 256), uint8."""
    
    try:
        mask = np.array(COLOR_MAP[color.lower()], dtype=np.uint8)[:, None]
    except KeyError:
        raise ValueError(f"Unsupported LUT color: {color}")

    return mask * np.arange(256, dtype=np.uint8)   