import numpy as np
import pytest
import surfa as sf
from surfa.mesh.robust_intersection import self_intersecting_face_pairs


def _sphere(subdivisions=3):
    """Unit sphere from a subdivided octahedron (closed, intersection-free)."""
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for _ in range(subdivisions):
        vertices = list(vertices)
        midpoint = {}

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoint:
                midpoint[key] = len(vertices)
                vertices.append((vertices[i] + vertices[j]) / 2)
            return midpoint[key]
        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        vertices = np.asarray(vertices)
        faces = np.asarray(new)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices, faces


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_separated_and_crossing_triangles():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0.2, 0.2, -0.5], [0.2, 0.2, 0.5], [0.8, 0.8, 0.1]], dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    assert self_intersecting_face_pairs(v, f).tolist() == [[0, 1]]
    v_far = v.copy()
    v_far[3:, 2] += 2.0
    assert len(self_intersecting_face_pairs(v_far, f)) == 0


def test_contact_is_not_an_intersection():
    # a vertex of the second triangle touches the interior of the first, the rest lies above it
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0.25, 0.25, 0.0], [0.25, 0.25, 1.0], [0.6, 0.1, 1.0]], dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    assert len(self_intersecting_face_pairs(v, f)) == 0


def test_shared_vertex_fold_is_found():
    # two faces share vertex 0; the second folds back through the first
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, -0.5], [0.3, 0.3, 0.5]], dtype=float)
    f = np.array([[0, 1, 2], [0, 3, 4]])
    assert self_intersecting_face_pairs(v, f).tolist() == [[0, 1]]


def test_adjacent_faces_are_ignored():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    assert len(self_intersecting_face_pairs(v, f)) == 0


def test_clean_sphere_under_transforms_and_round_off():
    v, f = _sphere(4)
    rng = np.random.default_rng(0)
    variants = [v, v @ _rotation(1).T, v + 1000.0, v * 1e-3, v * 1e3,
                v + 1e-9 * rng.normal(size=v.shape), v.astype(np.float32).astype(np.float64)]
    for variant in variants:
        assert len(self_intersecting_face_pairs(variant, f)) == 0


def test_planted_intersection_is_invariant():
    v, f = _sphere(4)
    v = v.copy()
    tip = np.argmax(v[:, 2])
    v[tip] = [0.013, 0.021, -1.3]      # the north pole is pulled through the south pole (off-axis: no degenerate hits)
    reference = self_intersecting_face_pairs(v, f)
    assert len(reference) > 0
    R = _rotation(2)
    for variant in (v @ R.T, v + 500.0, 0.01 * v, 100.0 * v, v + 1e-9 * np.random.default_rng(3).normal(size=v.shape)):
        assert np.array_equal(self_intersecting_face_pairs(variant, f), reference)


def test_mesh_method():
    v, f = _sphere(3)
    v = v.copy()
    v[np.argmax(v[:, 2])] = [0.013, 0.021, -1.3]
    mesh = sf.Mesh(v, f)
    faces, pairs = mesh.find_self_intersecting_faces_robust(return_pairs=True)
    assert np.array_equal(faces, np.unique(pairs))
    overlay = mesh.find_self_intersecting_faces_robust(overlay=True)
    assert overlay.shape[0] == mesh.nfaces
    assert np.array_equal(np.flatnonzero(np.asarray(overlay.data).ravel()), faces)
    assert len(sf.Mesh(*_sphere(3)).find_self_intersecting_faces_robust()) == 0


def test_broad_phase_misses_nothing():
    from surfa.mesh.robust_intersection import _pairs_intersect
    v, f = _sphere(2)
    v = v.copy()
    v[np.argmax(v[:, 2])] = [0.013, 0.021, -1.3]
    v += 0.02 * np.random.default_rng(4).normal(size=v.shape)
    i, j = np.triu_indices(len(f), k=1)
    every = np.stack([i, j], axis=1)
    brute = every[_pairs_intersect(v[f], f.astype(np.int64), every, 1e-10)]
    assert len(brute) > 0
    assert np.array_equal(self_intersecting_face_pairs(v, f), brute)
