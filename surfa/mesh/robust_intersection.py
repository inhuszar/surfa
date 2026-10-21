import numpy as np


def self_intersecting_face_pairs(vertices, faces, tolerance=1e-10, chunk_size=500000):
    """
    Find pairs of mesh faces that intersect each other, with a test that does not depend on round-off.

    Every face pair whose bounding boxes overlap is tested (found with a uniform grid), so no intersection
    is missed for lack of neighbors. Two faces intersect when an edge of one properly crosses the other: its endpoints lie
    strictly on opposite sides of the other face's plane, and the crossing point lies strictly inside
    the other face. All predicates are signed volumes computed in double precision relative to a vertex
    of the pair, and a predicate counts as zero when its magnitude is below `tolerance * L^3`, where L is
    the longest edge of the pair. The result is therefore unchanged by translation, rotation and uniform
    scaling of the mesh and by perturbations far below `tolerance * L`.

    Faces that share an edge are never reported. Faces that share one vertex are reported when the edge
    of either face opposite the shared vertex crosses the other face. Contacts (a vertex or an edge
    touching another face, to within the tolerance) and coplanar overlaps are not intersections here.

    Parameters
    ----------
    vertices : float (V, 3)
        Mesh vertex array.
    faces : int (F, 3)
        Mesh face array.
    tolerance : float
        Relative tolerance below which a signed volume counts as zero. A contact closer than about
        `tolerance * L` is not reported.
    chunk_size : int
        Number of candidate pairs tested at once (bounds the memory use).

    Returns
    -------
    pairs : int array (P, 2)
        Indices of intersecting face pairs, each pair sorted, rows in lexicographic order.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'expected faces of shape (F, 3), got {faces.shape}')
    if faces.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)

    # candidate pairs: faces whose axis-aligned bounding boxes overlap
    triangles = vertices[faces]
    candidates = _overlapping_boxes(triangles.min(1), triangles.max(1))
    if candidates.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    found = []
    for start in range(0, candidates.shape[0], chunk_size):
        pairs = candidates[start:start + chunk_size]
        hit = _pairs_intersect(triangles, faces, pairs, tolerance)
        found.append(pairs[hit])
    found = np.sort(np.concatenate(found), axis=1)
    return found[np.lexsort((found[:, 1], found[:, 0]))]


def _overlapping_boxes(lo, hi):
    """
    All pairs (i, j), i < j, of boxes [lo, hi] (N, 3) that overlap, found by hashing the boxes into a uniform
    grid whose cell is the median box extent.
    """
    n = lo.shape[0]
    extent = (hi - lo).max(1)
    cell = np.median(extent)
    if not cell > 0:
        cell = max(extent.max(), 1.0)
    origin = lo.min(0)
    first = np.floor((lo - origin) / cell).astype(np.int64)
    last = np.floor((hi - origin) / cell).astype(np.int64)
    span = last - first + 1
    count = span.prod(1)

    # one entry per (box, grid cell it touches)
    box = np.repeat(np.arange(n), count)
    local = np.arange(box.size) - np.repeat(np.cumsum(count) - count, count)
    sy, sz = span[box, 1], span[box, 2]
    offset = np.stack([local // (sy * sz), (local // sz) % sy, local % sz], axis=1)
    index = first[box] + offset
    dims = last.max(0) + 1
    key = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    order = np.argsort(key, kind='stable')
    key, box = key[order], box[order]

    # every pair of entries in the same cell
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, key.size])
    end = np.repeat(starts + sizes, sizes)
    position = np.arange(key.size)
    after = end - position - 1
    first_of = np.repeat(position, after)
    second_of = first_of + 1 + (np.arange(first_of.size) - np.repeat(np.cumsum(after) - after, after))
    a, b = box[first_of], box[second_of]
    a, b = np.minimum(a, b), np.maximum(a, b)
    keep = a != b
    pairs = np.unique(a[keep] * n + b[keep])
    pairs = np.stack([pairs // n, pairs % n], axis=1)

    # exact box overlap
    overlap = np.all((lo[pairs[:, 0]] <= hi[pairs[:, 1]]) & (lo[pairs[:, 1]] <= hi[pairs[:, 0]]), axis=1)
    return pairs[overlap]


def self_intersecting_faces_mask(vertices, faces, tolerance=1e-10, chunk_size=500000):
    """
    Boolean mask (F,) of the faces that take part in any intersection found by
    `self_intersecting_face_pairs` (same parameters).
    """
    pairs = self_intersecting_face_pairs(vertices, faces, tolerance=tolerance, chunk_size=chunk_size)
    mask = np.zeros(len(faces), dtype=bool)
    mask[pairs.ravel()] = True
    return mask


def _orient(a, b, c, d):
    """Signed volume det[b - a, c - a, d - a] for stacked points (..., 3)."""
    return np.einsum('...i,...i->...', np.cross(b - a, c - a), d - a)


def _crosses(p, q, a, b, c, tol):
    """Whether segment pq properly crosses triangle abc (all (N, 3), relative to a common origin)."""
    op = _orient(a, b, c, p)
    oq = _orient(a, b, c, q)
    straddles = ((op > tol) & (oq < -tol)) | ((op < -tol) & (oq > tol))
    s0 = _orient(p, q, a, b)
    s1 = _orient(p, q, b, c)
    s2 = _orient(p, q, c, a)
    inside = ((s0 > tol) & (s1 > tol) & (s2 > tol)) | ((s0 < -tol) & (s1 < -tol) & (s2 < -tol))
    return straddles & inside


def _pairs_intersect(triangles, faces, pairs, tolerance):
    """Intersection test for candidate face pairs (N, 2) -> bool (N,)."""
    fa = faces[pairs[:, 0]]
    fb = faces[pairs[:, 1]]
    shared = (fa[:, :, None] == fb[:, None, :])          # (N, 3, 3): vertex i of a equals vertex j of b
    nshared = shared.sum((1, 2))

    # coordinates relative to the first vertex of the first face: translation-invariant round-off
    origin = triangles[pairs[:, 0], 0]
    A = triangles[pairs[:, 0]] - origin[:, None]
    B = triangles[pairs[:, 1]] - origin[:, None]

    # the tolerance on signed volumes scales with the cube of the longest edge of the pair
    edges = np.concatenate([A - np.roll(A, 1, axis=1), B - np.roll(B, 1, axis=1)], axis=1)
    scale = np.linalg.norm(edges, axis=-1).max(-1)
    tol = tolerance * scale ** 3

    hit = np.zeros(len(pairs), dtype=bool)

    # no shared vertex: some edge of either face properly crosses the other face
    sel = np.flatnonzero(nshared == 0)
    if sel.size:
        a, b, t = A[sel], B[sel], tol[sel]
        h = np.zeros(sel.size, dtype=bool)
        for i in range(3):
            j = (i + 1) % 3
            h |= _crosses(a[:, i], a[:, j], b[:, 0], b[:, 1], b[:, 2], t)
            h |= _crosses(b[:, i], b[:, j], a[:, 0], a[:, 1], a[:, 2], t)
        hit[sel] = h

    # one shared vertex: the edge of either face opposite the shared vertex crosses the other face
    sel = np.flatnonzero(nshared == 1)
    if sel.size:
        a, b, t = A[sel], B[sel], tol[sel]
        ia = shared[sel].any(2).argmax(1)                # position of the shared vertex in face a
        ib = shared[sel].any(1).argmax(1)                # and in face b
        rows = np.arange(sel.size)
        pa, qa = a[rows, (ia + 1) % 3], a[rows, (ia + 2) % 3]
        pb, qb = b[rows, (ib + 1) % 3], b[rows, (ib + 2) % 3]
        hit[sel] = (_crosses(pa, qa, b[:, 0], b[:, 1], b[:, 2], t) |
                    _crosses(pb, qb, a[:, 0], a[:, 1], a[:, 2], t))

    # two or three shared vertices (adjacent or duplicate faces) are never reported
    return hit
