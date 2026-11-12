import os
import subprocess
import numpy as np
import pytest
import surfa as sf


FREESURFER_HOME = '/usr/local/freesurfer/8.2.0'
if not os.path.isdir(FREESURFER_HOME):
    FREESURFER_HOME = os.environ.get('FREESURFER_HOME', '')
LTA_CONVERT = os.path.join(FREESURFER_HOME, 'bin', 'lta_convert')
HAVE_LTA_CONVERT = bool(FREESURFER_HOME) and os.access(LTA_CONVERT, os.X_OK)

DATA_DIR = '/autofs/space/mirabel_001/users/ih967/tirl_example_data/formats-added-20261112/fsl'
REFERENCE = os.path.join(DATA_DIR, 'generated-reference_MNI152_T1_2mm_brain.nii.gz')
MOVING = os.path.join(DATA_DIR, 'generated-moving_MNI152_T1_2mm_skull.nii.gz')
HAVE_DATA = os.path.isfile(REFERENCE) and os.path.isfile(MOVING)

# register.dat as written by lta_convert --outreg (FreeSurfer 8.2.0)
REGISTER_DAT = """subject-unknown
2.000000
2.000000
0.100000
9.403766393661499e-01 6.173301115632057e-02 1.378990858793259e-01 -3.659210205078125e-01
-5.130974948406219e-02 9.753229022026062e-01 -8.532969653606415e-02 -8.114837646484375e+00
-1.443069428205490e-01 1.010028421878815e-01 1.015957832336426e+00 -3.019241333007812e+00
0 0 0 1
round
"""


def _known_affine():
    """Known scanner-RAS affine from the moving to the reference volume."""
    return np.asarray(sf.transform.compose_affine(
        translation=(3, -5, 7), rotation=(5, -3, 8), scale=(1.05, 0.97, 1.02),
        shear=(0.01, 0, 0.02)))


def _fs_env():
    env = os.environ.copy()
    env['FREESURFER_HOME'] = FREESURFER_HOME
    for name in ('license.txt', '.license'):
        license_file = os.path.join(FREESURFER_HOME, name)
        if os.path.isfile(license_file):
            env.setdefault('FS_LICENSE', license_file)
            break
    return env


def _lta_convert(*args):
    subprocess.run([LTA_CONVERT, *args], env=_fs_env(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture(params=['pair', 'oblique'])
def volumes(request, tmp_path):
    """
    Reference and moving volumes, plus the tolerance for comparing with lta_convert. The
    'pair' case uses the two volumes as they are, which share one geometry. The 'oblique'
    case replaces the moving volume by an oblique, cropped, anisotropic copy, so that
    source and target tkregister spaces differ and a swap of the two would show. Its
    larger tkregister offsets need a looser tolerance, because lta_convert computes in
    single precision (FreeSurfer's MATRIX is float).
    """
    ref = sf.load_volume(REFERENCE)
    mov = sf.load_volume(MOVING)
    if request.param == 'pair':
        return ref, mov, REFERENCE, MOVING, 1e-5
    data = mov.data[5:80, 10:100, 3:85]
    rotation = np.asarray(sf.transform.affine.angles_to_rotation_matrix((4, -6, 10)))
    geom = sf.ImageGeometry(data.shape, voxsize=(2.5, 2.2, 3.0),
                            rotation=rotation @ mov.geom.rotation, center=(4, -9, 12))
    mov = sf.Volume(data, geometry=geom)
    mov_file = str(tmp_path / 'moving.nii.gz')
    mov.save(mov_file)
    return ref, mov, REFERENCE, mov_file, 5e-5


def test_load_fields(tmp_path):
    """
    Header fields and the tkregister-space matrix of a register.dat file.
    """
    filename = tmp_path / 'register.dat'
    filename.write_text(REGISTER_DAT)
    aff = sf.load_affine(str(filename))
    assert aff.space == 'surf'
    assert aff.source is None and aff.target is None
    assert aff.metadata == {'subject': 'subject-unknown', 'inplaneres': 2.0,
                            'betplaneres': 2.0, 'intensity': 0.1, 'float2int': 'round'}
    R = np.loadtxt(str(filename), skiprows=4, max_rows=4)
    assert np.allclose(aff.inv().matrix, R, atol=1e-12, rtol=0)


def test_round_trip(tmp_path):
    """
    Load, save and load gives the same matrix and header fields, with or without a
    float2int line.
    """
    for text in (REGISTER_DAT, REGISTER_DAT.replace('round\n', ''),
                 REGISTER_DAT.replace('subject-unknown', 'bert').replace('round', 'tkregister')):
        first = tmp_path / 'first.reg'
        second = tmp_path / 'second.dat'
        first.write_text(text)
        a = sf.load_affine(str(first))
        a.save(str(second))
        b = sf.load_affine(str(second))
        assert b.space == 'surf'
        assert b.metadata == a.metadata
        assert np.allclose(a.matrix, b.matrix, atol=1e-12, rtol=0)
        assert first.read_text().split()[:4] == second.read_text().split()[:4]
        assert first.read_text().split()[20:] == second.read_text().split()[20:]


def test_format_detection(tmp_path):
    """
    Explicit format name, extension enforcement and rejection of non-register.dat files.
    """
    filename = tmp_path / 'register.dat'
    filename.write_text(REGISTER_DAT)
    aff = sf.load_affine(str(filename), fmt='regdat')
    sf.io.affine.save_affine(aff, str(tmp_path / 'noext'), fmt='regdat')
    assert (tmp_path / 'noext.dat').is_file()

    bad = tmp_path / 'table.dat'
    bad.write_text('1 2 3\n4 5 6\n')
    with pytest.raises(ValueError):
        sf.load_affine(str(bad))

    bad.write_text(REGISTER_DAT.replace('round', 'nearest'))
    with pytest.raises(ValueError):
        sf.load_affine(str(bad))


def test_save_requires_geometry(tmp_path):
    """
    A non-surface affine without geometries cannot be written.
    """
    aff = sf.Affine(_known_affine(), space='world')
    with pytest.raises(ValueError):
        aff.save(str(tmp_path / 'register.dat'))


@pytest.mark.skipif(not HAVE_DATA, reason='example volumes not found')
def test_scanner_ras_by_attaching_geometries(tmp_path, volumes):
    """
    A register.dat loaded without geometries converts to scanner RAS once the movable
    (source) and reference (target) geometries are attached.
    """
    ref, mov, _, _, _ = volumes
    known = sf.Affine(_known_affine(), source=mov, target=ref, space='world')
    filename = str(tmp_path / 'register.dat')
    known.save(filename)

    aff = sf.load_affine(filename)
    aff.source = mov
    aff.target = ref
    world = aff.convert(space='world')
    assert np.allclose(world.matrix, known.matrix, atol=1e-10, rtol=0)

    aff = sf.load_affine(filename, source=mov, target=ref)
    assert np.allclose(aff.convert(space='world').matrix, known.matrix, atol=1e-10, rtol=0)


@pytest.mark.skipif(not HAVE_LTA_CONVERT, reason='lta_convert not found')
@pytest.mark.skipif(not HAVE_DATA, reason='example volumes not found')
def test_read_lta_convert_outreg(tmp_path, volumes):
    """
    surfa reads the register.dat that lta_convert --outreg writes to the original
    transform.
    """
    ref, mov, ref_file, mov_file, tol = volumes
    known = sf.Affine(_known_affine(), source=mov, target=ref, space='world')
    lta = str(tmp_path / 'x.lta')
    dat = str(tmp_path / 'y.dat')
    known.save(lta)
    _lta_convert('--inlta', lta, '--outreg', dat, '--src', mov_file, '--trg', ref_file)

    aff = sf.load_affine(dat, source=mov, target=ref)
    world = aff.convert(space='world')
    assert np.allclose(world.matrix, known.matrix, atol=tol, rtol=0), np.abs(world.matrix - known.matrix).max()
    assert aff.metadata['float2int'] == 'round'
    assert np.isclose(aff.metadata['inplaneres'], mov.geom.voxsize[0])
    assert np.isclose(aff.metadata['betplaneres'], mov.geom.voxsize[2])


@pytest.mark.skipif(not HAVE_LTA_CONVERT, reason='lta_convert not found')
@pytest.mark.skipif(not HAVE_DATA, reason='example volumes not found')
def test_write_for_lta_convert_inreg(tmp_path, volumes):
    """
    lta_convert --inreg reads a register.dat written by surfa to the original transform.
    """
    ref, mov, ref_file, mov_file, tol = volumes
    known = sf.Affine(_known_affine(), source=mov, target=ref, space='world')
    dat = str(tmp_path / 'z.dat')
    lta = str(tmp_path / 'w.lta')
    known.save(dat)
    _lta_convert('--inreg', dat, '--outlta', lta, '--src', mov_file, '--trg', ref_file)

    back = sf.load_affine(lta)
    world = back.convert(space='world')
    assert np.allclose(world.matrix, known.matrix, atol=tol, rtol=0), np.abs(world.matrix - known.matrix).max()
