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

## Role in FITS

FITS decides which artifact a workflow step requires and where its result
belongs. `fits-io` performs the actual reads and writes and carries provenance,
axis information, calibration, and custom metadata across those operations.
The processing submodules therefore receive arrays rather than file paths.
