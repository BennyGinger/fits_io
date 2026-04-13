from fits_io.metadata.arrays import resolve_axes


def test_resolve_axes_prefers_explicit_axis_order():
    out = resolve_axes(axis_order='TCYX', reader_axes='TZCYX', z_projection=None, n_channels=2)
    assert out == 'TCYX'


def test_resolve_axes_drops_z_when_projected():
    out = resolve_axes(axis_order=None, reader_axes='TZCYX', z_projection='max', n_channels=2)
    assert out == 'TCYX'


def test_resolve_axes_drops_c_for_single_channel():
    out = resolve_axes(axis_order=None, reader_axes='TCYX', z_projection=None, n_channels=1)
    assert out == 'TYX'