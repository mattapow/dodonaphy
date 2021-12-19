import numpy as np
cimport numpy as np

cpdef nj_np(np.ndarray[np.double_t, ndim=2] pdm):
    cdef np.int_t leaf_node_count = len(pdm)
    cdef np.int_t node_count = 2 * leaf_node_count - 2
    cdef np.double_t eps = np.double(2.220446049250313e-16)

    cdef np.ndarray[np.int_t, ndim=2] peel = np.zeros((leaf_node_count - 1, 3), dtype=int)
    cdef np.ndarray[np.double_t, ndim=1] blens = np.zeros(node_count, dtype=np.double)

    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.array(leaf_node_count * [False])
    cdef np.ndarray[np.int_t, ndim=1] node_map = np.arange(leaf_node_count)

    cdef np.int_t left
    cdef np.int_t right
    cdef np.int_t parent
    cdef np.int_t int_i
    cdef np.ndarray[np.double_t, ndim=1] dist_p
    cdef np.double_t dist_pl
    cdef np.double_t dist_lr
    cdef np.int_t n = pdm.shape[0]
    cdef np.int_t m = pdm.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] Q = np.zeros_like(pdm)

    for int_i in range(leaf_node_count - 1):
        Q = compute_Q(pdm, mask)
        left, right = unravel_index(Q.argmin(), (n, m))
        dist_p = get_new_dist(pdm, mask, left, right)
        dist_pl = dist_p[left]
        dist_lr = pdm[left][right]

        parent = int_i + leaf_node_count
        peel[int_i, 0] = node_map[left]
        peel[int_i, 1] = node_map[right]
        peel[int_i, 2] = parent

        blens[node_map[left]] = dist_pl
        blens[node_map[right]] = np.maximum(dist_lr - dist_pl, eps)

        # replace right by dist_p in the pdm
        pdm = np.vstack((pdm[:right, :], dist_p, pdm[right + 1 :, :]))
        pdm = np.hstack(
            (pdm[:, :right], np.expand_dims(dist_p, -1), pdm[:, right + 1 :])
        )
        node_map[right] = parent
        mask[left] = True

    return peel, blens


cdef compute_Q(
    np.ndarray[np.double_t, ndim=2] pdm,
    mask=0,
    fill_value=None):
    """Compute the Q matrix for Neighbour joining.

    Args:
        pdm (ndarray): [description]
        n_active (long): [description]

    Returns:
        [type]: [description]
    """
    n_pdm = len(pdm)
    if mask is False:
        mask = np.full((n_pdm,), False)
    n_active = sum(~mask)

    cdef np.ndarray[np.uint8_t, ndim=1] mask_invert = np.invert(mask)
    cdef np.ndarray[np.uint8_t, ndim=2] mask_2d = np.invert(np.outer(mask_invert, mask_invert))
    cdef np.ndarray[np.double_t, ndim=2] sum_pdm = np.sum(pdm * ~mask_2d, axis=1, keepdims=True)
    cdef np.ndarray[np.double_t, ndim=2] sum_i = np.tile(sum_pdm, (1, n_pdm))
    cdef np.ndarray[np.double_t, ndim=2] sum_j = np.tile(sum_pdm.T, (n_pdm, 1))
    Q = np.ma.array(
        (n_active - 2) * pdm - sum_i - sum_j,
        mask=mask_2d
        )

    if fill_value is None:
        fill_value = np.finfo(np.double).max / 2.0
    np.ma.filled(Q, fill_value)
    np.fill_diagonal(Q, fill_value)
    Q = 0.5 * (Q + Q.T)
    return np.ma.getdata(Q)


cdef unravel_index(np.int_t index, shape):
    out = []
    cdef np.int_t dim
    for dim in reversed(shape):
        out.append(index % dim)
        index = np.floor(index / dim)
    return tuple(reversed(out))

cdef get_new_dist(
    np.ndarray[np.double_t, ndim=2] pdm,
    np.ndarray[np.uint8_t, ndim=1] mask,
    np.int_t f,
    np.int_t g):
    """Get neighbour joining distances to new node."""
    cdef np.double_t eps = np.double(2.220446049250313e-16)
    # get distance all taxa to new node u
    cdef np.ndarray[np.double_t, ndim=1] dist_u = 0.5 * (pdm[f] + pdm[g] - pdm[f][g])

    # get distance from u to f
    cdef np.ndarray[np.uint8_t, ndim=1] mask_invert = np.invert(mask)
    cdef np.int_t n_active = np.maximum(sum(mask_invert), 3)
    cdef np.ndarray[np.uint8_t, ndim=2] mask_invert_2d = np.outer(mask_invert, mask_invert)
    cdef np.ndarray[np.double_t, ndim=1] sum_pdm = np.sum(pdm * mask_invert_2d, axis=-1)
    cdef np.double_t dist_uf = 0.5 * pdm[f][g] + (sum_pdm[f] - sum_pdm[g]) / (2 * (n_active - 2))
    dist_u[f] = np.maximum(dist_uf, eps)
    dist_u[g] = 0.0
    return dist_u
