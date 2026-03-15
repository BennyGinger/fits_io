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