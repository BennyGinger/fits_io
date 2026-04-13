from fits_io.metadata.codec import FITS_TAG


def test_fits_tag_constant_is_stable():
    assert FITS_TAG == 65000
