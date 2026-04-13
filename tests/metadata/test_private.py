from fits_io.metadata.context import MetadataBuildContext
from fits_io.metadata.payload import build_private_payload


def _make_ctx() -> MetadataBuildContext:
    return MetadataBuildContext(
        n_channels=2,
        labels=['GFP', 'RFP'],
        axes='TZCYX',
        base_payload={'existing': 'data'},
        interval=11.0,
        resolution=(0.5, 0.25),
    )


def test_build_private_payload_writes_fits_io_block():
    ctx = _make_ctx()
    out = build_private_payload(ctx, z_projection='mean', compression='zlib')
    assert out['existing'] == 'data'
    assert out['fits_io']['axes'] == 'TZCYX'
    assert out['fits_io']['n_channels'] == 2
    assert out['fits_io']['z_projection'] == 'mean'
    assert out['fits_io']['compression'] == 'zlib'
    assert 'version' in out['fits_io']


def test_build_private_payload_sets_project_metadata_when_provided():
    ctx = _make_ctx()
    out = build_private_payload(ctx, project_metadata={'run_id': 42, 'notes': 'ok'})
    assert out['project_metadata'] == {'run_id': 42, 'notes': 'ok'}


def test_build_private_payload_does_not_set_project_metadata_when_missing():
    ctx = _make_ctx()
    out = build_private_payload(ctx)
    assert 'project_metadata' not in out


def test_build_private_payload_sets_source_channel_identity_when_provided():
    ctx = MetadataBuildContext(
        n_channels=2,
        labels=['GFP', 'RFP'],
        axes='TZCYX',
        base_payload={'existing': 'data'},
        interval=11.0,
        resolution=(0.5, 0.25),
        source_channel_indices=[1, 2],
        source_channel_count=3,
    )
    out = build_private_payload(ctx)
    assert out['fits_io']['source_channel_indices'] == [1, 2]
    assert out['fits_io']['source_channel_count'] == 3


def test_build_private_payload_preserves_existing_source_channel_identity_when_not_provided():
    ctx = MetadataBuildContext(
        n_channels=2,
        labels=['GFP', 'RFP'],
        axes='TZCYX',
        base_payload={
            'existing': 'data',
            'fits_io': {
                'source_channel_indices': [0, 2],
                'source_channel_count': 3,
            },
        },
        interval=11.0,
        resolution=(0.5, 0.25),
    )
    out = build_private_payload(ctx)
    assert out['fits_io']['source_channel_indices'] == [0, 2]
    assert out['fits_io']['source_channel_count'] == 3


def test_build_private_payload_preserves_unknown_existing_fits_io_keys():
    ctx = MetadataBuildContext(
        n_channels=2,
        labels=['GFP', 'RFP'],
        axes='TZCYX',
        base_payload={
            'fits_io': {
                'custom_key': 'keep-me',
                'source_channel_indices': [1, 2],
            }
        },
        interval=11.0,
        resolution=(0.5, 0.25),
    )
    out = build_private_payload(ctx)
    assert out['fits_io']['custom_key'] == 'keep-me'
    assert out['fits_io']['source_channel_indices'] == [1, 2]