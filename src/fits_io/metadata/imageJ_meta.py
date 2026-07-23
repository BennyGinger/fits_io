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

    _DELIMITER_CHARS = ("* ", "=", "+", "~", ". ")
    _DELIMITER_WIDTH = 50

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
    def _delimiter(cls, depth: int) -> str:
        """
        Return the delimiter associated with a zero-based nesting depth.

        Depths beyond the configured styles reuse the final delimiter style.
        """
        style_index = min(depth, len(cls._DELIMITER_CHARS) - 1)
        delimiter_char = cls._DELIMITER_CHARS[style_index]

        return delimiter_char * cls._DELIMITER_WIDTH

    @classmethod
    def _append_heading(cls,
                        lines: list[str],
                        *,
                        section: str,
                        depth: int,
                        ) -> None:
        delimiter = cls._delimiter(depth)

        lines.extend([delimiter,
                      section,
                      delimiter,
                      "",])

    @staticmethod
    def _partition_items(value: Mapping[str, Any],) -> tuple[list[tuple[str, Any]],
                                                             list[tuple[str, Mapping[str, Any]]],]:
        """
        Split a mapping into scalar values and nested mappings.
        """
        scalar_items: list[tuple[str, Any]] = []
        nested_items: list[tuple[str, Mapping[str, Any]]] = []

        for key, child in value.items():
            key = str(key)

            if isinstance(child, Mapping):
                nested_items.append((key, child))
            else:
                scalar_items.append((key, child))

        return scalar_items, nested_items

    @classmethod
    def _append_toml_sections(cls,
                              lines: list[str],
                              *,
                              section: str,
                              value: Mapping[str, Any],
                              depth: int,
                              delimiter_levels: int,
                              ) -> None:
        if not cls._has_renderable_content(value):
            return
        
        scalar_items, nested_items = cls._partition_items(value)

        # Public levels are one-based:
        # delimiter_levels=1 renders headings at internal depth 0.
        if depth < delimiter_levels:
            cls._append_heading(lines,
                                section=section,
                                depth=depth,)

        # Do not render an empty TOML section when this mapping only
        # contains nested mappings.
        if scalar_items:
            lines.append(f"[{section}]")

            for key, child in scalar_items:
                lines.append(f"{key} = {cls._toml_value(child)}")

            lines.append("")

        for key, child in nested_items:
            cls._append_toml_sections(lines,
                                      section=f"{section}.{key}",
                                      value=child,
                                      depth=depth + 1,
                                      delimiter_levels=delimiter_levels,)

    @classmethod
    def _has_renderable_content(cls, value: Mapping[str, Any],) -> bool:
        """
        Return whether a mapping contains at least one scalar value,
        directly or in a nested mapping.
        """
        for _, child in value.items():
            if isinstance(child, Mapping):
                if child and cls._has_renderable_content(child):
                    return True
            else:
                return True

        return False
    
    def render(self, *, delimiter_levels: int = 2,) -> str:
        """
        Render the payload as a TOML-like ImageJ metadata summary.

        ``delimiter_levels`` controls how many nesting levels receive a
        visual heading:

        - 0: no delimiter headings
        - 1: top-level sections only
        - 2: top-level and direct child sections
        - 3: include one additional nested level
        - 4: include one additional nested levels
        - 5: include the last additional nested levels
        
        Up to 5 levels are supported. Levels beyond 5 will only use the same format as 5.
        """
        if delimiter_levels < 0:
            raise ValueError("delimiter_levels must be non-negative.")

        if not self.payload:
            return ""

        lines: list[str] = []

        for key, value in self.payload.items():
            key = str(key)

            if isinstance(value, Mapping):
                self._append_toml_sections(
                    lines,
                    section=key,
                    value=value,
                    depth=0,
                    delimiter_levels=delimiter_levels,)
            else:
                lines.append(f"{key} = {self._toml_value(value)}")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"
   
            
def make_color_lut(color: str) -> NDArray[np.uint8]:
    """Return ImageJ-style LUT: shape (3, 256), uint8."""
    
    try:
        mask = np.array(COLOR_MAP[color.lower()], dtype=np.uint8)[:, None]
    except KeyError:
        raise ValueError(f"Unsupported LUT color: {color}")

    return mask * np.arange(256, dtype=np.uint8)   