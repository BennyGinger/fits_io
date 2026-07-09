from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.payload import assemble_payload


def test_build_payload_preserves_existing_source_channel_identity_when_not_overridden():
    base = FitsIOMeta().with_fitsio(
        source_channel_indices=[0, 2],
        source_channel_count=3,
    )

    out = assemble_payload(base)

    assert out.fits_io.source_channel_indices == [0, 2]
    assert out.fits_io.source_channel_count == 3


def test_build_payload_sets_source_channel_identity_when_provided():
    base = FitsIOMeta()

    out = assemble_payload(base, source_channel_indices=[1, 2], source_channel_count=3)

    assert out.fits_io.source_channel_indices == [1, 2]
    assert out.fits_io.source_channel_count == 3


def test_build_payload_keeps_unrelated_existing_fits_io_fields():
    base = FitsIOMeta().with_fitsio(axes="TZCYX", channel_labels=["GFP", "RFP"], n_channels=2)

    out = assemble_payload(base, compression="zlib")

    assert out.fits_io.axes == "TZCYX"
    assert out.fits_io.channel_labels == ["GFP", "RFP"]
    assert out.fits_io.n_channels == 2
    assert out.fits_io.compression == "zlib"
