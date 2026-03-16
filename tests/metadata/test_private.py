from fits_io.metadata.context import MetadataBuildContext
from fits_io.metadata.private import DEFAULT_STEP_NAME, _update_metadata, build_private_payload, get_status, get_step_name


def _make_ctx() -> MetadataBuildContext:
    return MetadataBuildContext(n_channels=2, labels=['GFP', 'RFP'], axes='TZCYX', base_payload={'existing': 'data'}, step_name='test_step', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))


def test_get_step_name_uses_provided_name():
    assert get_step_name({}, step_name='my_step') == 'my_step'


def test_get_step_name_default_and_increment():
    assert get_step_name({}, step_name=None) == DEFAULT_STEP_NAME
    assert get_step_name({'unknown_step_1': {}}, step_name=None) == 'unknown_step_2'


def test_get_status_falls_back_to_default_on_invalid():
    assert get_status({'status': 'active'}) == 'active'
    assert get_status({'status': 'invalid'}) == 'active'


def test_update_metadata_merges_into_existing_step():
    original = {'step1': {'a': 1}}
    out = _update_metadata(original, update_meta={'b': 2}, user_name='u', step_name='step1', z_projection='max', status='skip')
    assert out['step1'] == {'a': 1, 'b': 2}
    assert out['user_name'] == 'u'
    assert out['status'] == 'skip'
    assert out['z_projection_method'] == 'max'


def test_build_private_payload_without_provenance_still_updates_core_keys():
    ctx = _make_ctx()
    out = build_private_payload(ctx, add_step_meta=False, extra_step_metadata={'x': 1}, z_projection='mean')
    assert out['existing'] == 'data'
    assert out['status'] == 'active'
    assert out['user_name'] == 'test_user'
    assert out['z_projection_method'] == 'mean'
    assert out['test_step']['x'] == 1
    assert 'dist' not in out['test_step']


def test_build_private_payload_with_provenance_adds_profile_fields():
    ctx = _make_ctx()
    out = build_private_payload(ctx, distribution='test-dist', add_step_meta=True)
    assert out['status'] == 'active'
    assert out['user_name'] == 'test_user'
    assert 'test_step' in out
    assert 'dist' in out['test_step']
    assert 'version' in out['test_step']
    assert 'timestamp' in out['test_step']


def test_build_private_payload_creates_first_segment_channel_entry():
    ctx = MetadataBuildContext(n_channels=1, labels=['GFP'], axes='TYX', base_payload={'existing': 'data'}, step_name='segment', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))
    out = build_private_payload(ctx, distribution='test-dist', add_step_meta=True, extra_step_metadata={'channels': {'1': {'backend': 'v4'}}})
    assert out['segment']['channels'] == {'1': {'backend': 'v4'}}
    assert out['segment']['dist'] == 'test-dist'
    assert out['segment']['version']
    assert out['segment']['timestamp']


def test_build_private_payload_preserves_existing_segment_channels_when_adding_new_one():
    ctx = MetadataBuildContext(n_channels=1, labels=['RFP'], axes='TYX', base_payload={'segment': {'channels': {'1': {'model_name': 'old-gfp'}}, 'legacy': 'keep-me'}}, step_name='segment', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))
    out = build_private_payload(ctx, distribution='test-dist', add_step_meta=True, extra_step_metadata={'channels': {'2': {'model_name': 'new-rfp'}}})
    assert out['segment']['channels'] == {'1': {'model_name': 'old-gfp'}, '2': {'model_name': 'new-rfp'}}
    assert out['segment']['legacy'] == 'keep-me'


def test_build_private_payload_updates_only_requested_segment_channel():
    ctx = MetadataBuildContext(n_channels=1, labels=['GFP'], axes='TYX', base_payload={'segment': {'channels': {'1': {'model_name': 'old-gfp'}, '2': {'model_name': 'keep-rfp'}}}}, step_name='segment', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))
    out = build_private_payload(ctx, distribution='test-dist', add_step_meta=True, extra_step_metadata={'channels': {'1': {'model_name': 'new-gfp'}}})
    assert out['segment']['channels'] == {'1': {'model_name': 'new-gfp'}, '2': {'model_name': 'keep-rfp'}}


def test_build_private_payload_sets_source_channel_identity_when_provided():
    ctx = MetadataBuildContext(n_channels=2, labels=['GFP', 'RFP'], axes='TZCYX', base_payload={'existing': 'data'}, step_name='test_step', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25), source_channel_indices=[1, 2], source_channel_count=3)
    out = build_private_payload(ctx, add_step_meta=False)
    assert out['source_channel_indices'] == [1, 2]
    assert out['source_channel_count'] == 3


def test_build_private_payload_preserves_existing_source_channel_identity_when_not_provided():
    ctx = MetadataBuildContext(n_channels=2, labels=['GFP', 'RFP'], axes='TZCYX', base_payload={'existing': 'data', 'source_channel_indices': [0, 2], 'source_channel_count': 3}, step_name='test_step', status='active', user_name='test_user', interval=11.0, resolution=(0.5, 0.25))
    out = build_private_payload(ctx, add_step_meta=False)
    assert out['source_channel_indices'] == [0, 2]
    assert out['source_channel_count'] == 3