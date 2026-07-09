from fits_io.metadata.models import FitsIOMeta
from fits_io.metadata.payload import assemble_payload


def test_build_payload_sets_requested_fits_io_fields():
    base = FitsIOMeta()
    out = assemble_payload(
        base,
        axes="TCYX",
        channel_labels=["GFP", "RFP"],
        n_channels=2,
        source_channel_indices=[0, 2],
        source_channel_count=3,
        z_projection="max",
        compression="zlib",
    )

    assert out.fits_io.axes == "TCYX"
    assert out.fits_io.channel_labels == ["GFP", "RFP"]
    assert out.fits_io.n_channels == 2
    assert out.fits_io.source_channel_indices == [0, 2]
    assert out.fits_io.source_channel_count == 3
    assert out.fits_io.z_projection == "max"
    assert out.fits_io.compression == "zlib"


def test_build_payload_merges_custom_metadata_when_provided():
    base = FitsIOMeta(custom_metadata={"legacy": 1})
    out = assemble_payload(base, custom_metadata={"run_id": 7})

    assert out.custom_metadata == {"legacy": 1, "run_id": 7}


def test_build_payload_preserves_base_custom_metadata_when_custom_missing():
    base = FitsIOMeta(custom_metadata={"legacy": "keep"})
    out = assemble_payload(base)

    assert out.custom_metadata == {"legacy": "keep"}
