from typing import Any, Literal, Sequence, TypeAlias, get_args

# private tag to save custom metadata in tiff files
TiffTag: TypeAlias = tuple[
    int,    # tag id
    str,    # TIFF dtype
    int,    # count (0 = infer)
    Any,    # value
    bool,   # writeonce
]

ExtraTags: TypeAlias = Sequence[TiffTag]

PixelSize: TypeAlias = tuple[float, float]  # (x_um_per_pix, y_um_per_pix)
PixelDensity: TypeAlias = tuple[float, float]  # (x_pix_per_unit, y_pix_per_unit)

ExtTags = Literal['.tiff', '.tif', '.nd2']
SUPPORTED_EXTENSIONS: set[ExtTags] = set(get_args(ExtTags))

Zproj = Literal['max', 'mean', None]

ArrAxis = Literal['P', 'C', 'Z', 'T', 'X', 'Y']