"""Source display colors and optional conversion label defaults."""

from collections import Counter
from collections.abc import Sequence

from fits_io.metadata.imageJ_meta import COLOR_MAP


def channel_color_name(color: str | None) -> str | None:
    """
    Name a stored RGB color using the nearest normalized display palette color.
    """
    if color is None:
        return None
    rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    maximum = max(rgb)
    if not maximum:
        return None
    normalized = tuple(value / maximum for value in rgb)
    name = min(COLOR_MAP, key=lambda key: sum((value - target) ** 2
                                             for value, target in zip(normalized, COLOR_MAP[key])))
    return "white" if name == "gray" else name


def conversion_channel_labels(labels: str | Sequence[str] | None,
                               count: int,
                               colors: Sequence[str | None] | None = None,
                               ) -> list[str]:
    """
    Fill blank labels from colors, numbering repeated names; ignore colors on count conflicts.
    """
    entries = [labels] if isinstance(labels, str) else list(labels or [])
    if entries and len(entries) != count:
        return [f"C{index + 1}" for index in range(count)]
    entries = entries or [""] * count
    available = colors if colors is not None and len(colors) == count else [None] * count
    names = [channel_color_name(color) for color in available]
    totals = Counter(name for label, name in zip(entries, names) if not label.strip() and name)
    seen: Counter[str] = Counter()
    reserved = {label.strip() for label in entries if label.strip()}
    resolved: list[str] = []
    for index, (label, name) in enumerate(zip(entries, names)):
        label = label.strip()
        if not label:
            if name:
                seen[name] += 1
                label = f"{name}_{seen[name]}" if totals[name] > 1 else name
            else:
                label = f"C{index + 1}"
            base = label
            suffix = 1
            while label in reserved or label in resolved:
                label = f"{base}_{suffix}"
                suffix += 1
        resolved.append(label)
    return resolved
