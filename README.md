# fits-io

`fits-io` is the image I/O and artifact-format layer for FITS. It presents ND2
and TIFF microscopy data through one `FitsIO` façade, keeps axis and channel
identity explicit, and writes ImageJ-compatible TIFF files with structured FITS
metadata embedded in them.

```python
from fits_io import FitsIO

image = FitsIO.from_path("experiment.nd2")
result = image.get_channel("GFP", z_projection="max")

print(result.array.shape, result.axes)
```

The façade supports full-array and channel reads, series splitting, Z
projection, channel selection and reconstruction, channel-aware mask merging,
conversion preparation, and artifact saving. Channel labels are always tied to
their original source indices so that derived files remain traceable even when
only a subset of channels is exported.

Channel labels determine ImageJ LUT colours, for example GFP is green and DAPI
is blue. Unrecognised labels use grayscale without suppressing known colours
in other channels. Single-channel TIFFs also carry a TIFF colour map so ImageJ
restores their LUT when opening them, including files with embedded overlays.

`save_array(..., crop_xy=(x0, y0, x1, y1))` applies an exclusive XY pixel crop
across every other dimension, using the array's axes metadata. It preserves
channel identity, FITS/custom metadata, pixel calibration, and timing, and adds
the source shape, bounds, and offset to `spatial_crops` metadata. ImageJ spatial
origins are adjusted for the crop. Serialized annotations can be supplied with
`imagej_overlays`; the caller must first clip/shift them into crop coordinates.
Use `compression="none"` for overlay hyperstacks opened with ImageJ's native
TIFF opener.

## Role in FITS

FITS decides which artifact a workflow step requires and where its result
belongs. `fits-io` performs the actual reads and writes and carries provenance,
axis information, calibration, and custom metadata across those operations.
The processing submodules therefore receive arrays rather than file paths.
