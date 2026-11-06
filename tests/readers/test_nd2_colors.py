from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nd2
from nd2.structures import Color
import pytest

from fits_io.readers.r_nd2 import Nd2Reader


@pytest.mark.parametrize("colors, expected", [
    ([Color(0, 255, 0), Color(254, 0, 0)], ["#00ff00", "#fe0000"]),
    ([Color(255, 255, 255), Color(0, 0, 0)], ["#ffffff", None]),
    ([None, Color(0, 0, 255)], [None, "#0000ff"]),
    ([Color(0, 255, 0)], None),
    (None, None),
])
def test_channel_colors_are_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          fake_nd2_file_basic: Any,
                                          colors: list[Color | None] | None,
                                          expected: list[str | None] | None,
                                          ) -> None:
    """
    Preserve stored RGB values and reject missing, black or misaligned color metadata.
    """
    monkeypatch.setattr(nd2, "ND2File", fake_nd2_file_basic)
    reader = Nd2Reader(tmp_path / "input.nd2")
    monkeypatch.setattr(reader, "_sizes", {"C": 2, "Y": 8, "X": 8})
    channels = None if colors is None else [SimpleNamespace(channel=SimpleNamespace(color=c)) for c in colors]
    monkeypatch.setattr(reader, "_channels", channels)
    assert reader.channel_colors == expected
