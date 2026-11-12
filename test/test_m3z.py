import os
import shutil
import subprocess

import numpy as np
import pytest
import surfa as sf
from surfa.transform.geometry import image_geometry_equal


FREESURFER_HOME = '/usr/local/freesurfer/8.2.0'
MRI_WARP_CONVERT = os.path.join(FREESURFER_HOME, 'bin', 'mri_warp_convert')
REFERENCE = ('/autofs/space/mirabel_001/users/ih967/tirl_example_data/'
             'formats-added-20261112/fsl/generated-reference_MNI152_T1_2mm_brain.nii.gz')
COARSE_M3Z = os.path.join(FREESURFER_HOME, 'subjects', 'cvs_avg35', 'mri', 'transforms',
                          'talairach.m3z')

requires_fs = pytest.mark.skipif(not os.access(MRI_WARP_CONVERT, os.X_OK),
                                 reason='mri_warp_convert not found')

# float32 precision of absolute voxel coordinates up to ~100
TOL = 1e-4


def _source_geometry():
    """Source geometry of a corpus volume, or a stand-in with the same layout."""
    if os.path.exists(REFERENCE):
        return sf.load_volume(REFERENCE).geom
    return sf.ImageGeometry(shape=(91, 109, 91), voxsize=2,
                            vox2world=[[-2, 0, 0, 90], [0, 2, 0, -126],
                                       [0, 0, 2, -72], [0, 0, 0, 1]])


def _synthetic_warp():
    """Smooth disp_ras warp on a small, shifted target grid."""
    source = _source_geometry()
    target = sf.ImageGeometry(shape=(12, 14, 10), voxsize=4, rotation=source.rotation,
                              center=source.center + np.array([2.0, -3.0, 5.0]))
    u = np.meshgrid(*(np.linspace(0, 1, n) for n in target.shape), indexing='ij')
    disp = np.stack([3.0 * np.sin(np.pi * u[0]) * np.cos(np.pi * u[1]),
                     2.0 * np.sin(np.pi * u[1] * u[2]),
                     -2.5 * np.cos(np.pi * u[2]) * u[0]], axis=-1)
    return sf.Warp(disp.astype(np.float32), source=source, target=target,
                   format=sf.Warp.Format.disp_ras)


def _run(*args):
    env = dict(os.environ, FREESURFER_HOME=FREESURFER_HOME)
    license = os.path.join(FREESURFER_HOME, 'license.txt')
    if os.path.exists(license):
        env['FS_LICENSE'] = license
    result = subprocess.run([MRI_WARP_CONVERT, *args], env=env, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_same_warp(a, b, tol=TOL):
    a = a.convert(sf.Warp.Format.abs_crs)
    b = b.convert(sf.Warp.Format.abs_crs)
    assert a.shape == b.shape
    assert np.abs(a.data - b.data).max() <= tol
    assert image_geometry_equal(a.source, b.source, tol=1e-5)
    assert image_geometry_equal(a.target, b.target, tol=1e-5)


def test_m3z_round_trip(tmp_path):
    """
    Saving and loading an m3z gives the same field and geometries, and loading and
    saving again reproduces the field exactly.
    """
    warp = _synthetic_warp()
    warp.save(tmp_path / 'a.m3z')
    loaded = sf.load_warp(tmp_path / 'a.m3z')
    assert loaded.format == sf.Warp.Format.abs_crs
    _assert_same_warp(warp, loaded)

    loaded.save(tmp_path / 'b.m3z')
    again = sf.load_warp(tmp_path / 'b.m3z')
    assert again.format == sf.Warp.Format.abs_crs
    assert np.array_equal(again.data, loaded.data)
    assert image_geometry_equal(again.source, loaded.source)
    assert image_geometry_equal(again.target, loaded.target)
    for key in ('exp_k', 'source-valid', 'target-valid'):
        assert again.metadata[key] == loaded.metadata[key]
    assert np.array_equal(again.metadata['m3z-orig'], loaded.metadata['m3z-orig'])


def test_m3z_metadata_round_trip(tmp_path):
    """
    Node labels, original positions, exp_k and the linear transform survive a round trip.
    """
    warp = _synthetic_warp().convert(sf.Warp.Format.abs_crs)
    rng = np.random.default_rng(0)
    warp.metadata['m3z-labels'] = rng.integers(0, 50, warp.baseshape).astype(np.int32)
    warp.metadata['m3z-orig'] = rng.normal(size=warp.shape).astype(np.float32)
    warp.metadata['m3z-affine'] = np.array([[1.05, -0.1, 0.0, 2.5], [0.1, 1.1, 0.0, -17.0],
                                            [0.0, 0.0, 0.95, -8.9], [0, 0, 0, 1]])
    warp.metadata['exp_k'] = 20.0
    warp.save(tmp_path / 'a.m3z')

    loaded = sf.load_warp(tmp_path / 'a.m3z')
    assert np.array_equal(loaded.data, warp.data)
    for key in ('m3z-labels', 'm3z-orig'):
        assert np.array_equal(loaded.metadata[key], warp.metadata[key])
    assert np.allclose(loaded.metadata['m3z-affine'], warp.metadata['m3z-affine'],
                       atol=1e-6)
    assert loaded.metadata['exp_k'] == 20.0


@requires_fs
def test_m3z_from_mri_warp_convert(tmp_path):
    """
    An m3z written by mri_warp_convert from a surfa mgz warp loads as the same warp.
    """
    warp = _synthetic_warp()
    warp.save(tmp_path / 'a.mgz')
    _run('--inmgzwarp', str(tmp_path / 'a.mgz'), '--outm3z', str(tmp_path / 'a.m3z'))
    _assert_same_warp(warp, sf.load_warp(tmp_path / 'a.m3z'))


@requires_fs
def test_m3z_to_mri_warp_convert(tmp_path):
    """
    An m3z written by surfa converts with mri_warp_convert to an mgz holding the same
    warp.
    """
    warp = _synthetic_warp()
    warp.save(tmp_path / 'a.m3z')
    _run('--inm3z', str(tmp_path / 'a.m3z'), '--outmgzwarp', str(tmp_path / 'a.mgz'))
    _assert_same_warp(warp, sf.load_warp(tmp_path / 'a.mgz'))


@requires_fs
@pytest.mark.skipif(not os.path.exists(COARSE_M3Z), reason='cvs_avg35 m3z not found')
def test_m3z_coarse_node_grid(tmp_path):
    """
    A recon-all morph with node spacing 2 keeps its 1 mm atlas geometry when saved
    again, and mri_warp_convert reads the copy exactly as it reads the original.
    """
    warp = sf.load_warp(COARSE_M3Z)
    assert warp.metadata['m3z-spacing'] == 2
    assert tuple(warp.metadata['m3z-atlas'].shape) == (256, 256, 256)
    assert tuple(warp.target.shape) == tuple(warp.baseshape)
    assert np.allclose(warp.target.voxsize, 2 * warp.metadata['m3z-atlas'].voxsize)
    assert 'm3z-affine' in warp.metadata

    warp.save(tmp_path / 'copy.m3z')
    _run('--inm3z', COARSE_M3Z, '--outmgzwarp', str(tmp_path / 'orig.mgz'))
    _run('--inm3z', str(tmp_path / 'copy.m3z'), '--outmgzwarp', str(tmp_path / 'copy.mgz'))
    orig = sf.load_warp(tmp_path / 'orig.mgz')
    copy = sf.load_warp(tmp_path / 'copy.mgz')
    assert np.array_equal(orig.data, copy.data)
    assert image_geometry_equal(orig.source, copy.source)
    assert image_geometry_equal(orig.target, copy.target)
