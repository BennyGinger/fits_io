from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.payload import assemble_payload


def test_build_payload_preserves_existing_source_channel_identity_when_not_overridden():
    base = FitsIOMeta().with_fitsio(
        source_channel_indices=[0, 2],
        artifact_channel_indices=[0, 2],
    )

    out = assemble_payload(base)

    assert out.fits_io.source_channel_indices == [0, 2]
    assert out.fits_io.artifact_channel_indices == [0, 2]


def test_build_payload_sets_source_channel_identity_when_provided():
    base = FitsIOMeta()

    out = assemble_payload(
        base,
        source_channel_indices=[0, 1, 2],
        artifact_channel_indices=[1, 2],)

    assert out.fits_io.source_channel_indices == [0, 1, 2]
    assert out.fits_io.artifact_channel_indices == [1, 2]


def test_build_payload_keeps_unrelated_existing_fits_io_fields():
    base = FitsIOMeta().with_fitsio(
        axes="TZCYX", channel_labels=["GFP", "RFP"])

    out = assemble_payload(base, artifact_type="image")

    assert out.fits_io.axes == "TZCYX"
    assert out.fits_io.channel_labels == ["GFP", "RFP"]
    assert out.fits_io.channel_count == 2
    assert out.fits_io.artifact_type == "image"
