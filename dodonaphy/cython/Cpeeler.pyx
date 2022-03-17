from heapq import heapify, heappop, heappush

import numpy as np
cimport numpy as np
cimport cython

from dodonaphy.node import Node
from dodonaphy.edge import Edge


eps = np.finfo(np.double).eps

def nj_np(double[:, ::1] pdm):
    """Calculate neighbour joining tree.
    Credit to Dendropy for python implentation.

    Args:
        pdm (ndarray): Pairwise distance matrix
    """

    cdef int n_pool = len(pdm)
    cdef int n_taxa = len(pdm)
    cdef int n_ints = n_taxa - 1
    cdef int node_count = 2 * n_taxa - 2

    cdef np.ndarray[long, ndim=2] peel = np.zeros((n_ints, 3), dtype=int)
    cdef np.ndarray[np.double_t, ndim=1] blens = np.zeros(node_count, dtype=np.double)

    # initialise node pool
    node_pool = [Node(taxon=taxon) for taxon in range(n_pool)]

    cdef np.double_t dist
    cdef np.double_t v1
    cdef np.double_t v3
    cdef np.double_t v4
    cdef np.double_t qvalue
    cdef np.double_t min_q
    cdef int int_i
    cdef int parent
    cdef np.double_t delta_f
    cdef np.double_t delta_g

    # cache calculations
    for nd1 in node_pool:
        nd1._nj_xsub = 0.0
        for nd2 in node_pool:
            if nd1 is nd2:
                continue
            dist = pdm[nd1.taxon, nd2.taxon]
            nd1._nj_distances[nd2] = dist
            nd1._nj_xsub += dist

    while n_pool > 1:
        # calculate argmin of Q-matrix
        min_q = np.inf
        n_pool = len(node_pool)
        for idx1, nd1 in enumerate(node_pool[:-1]):
            for _, nd2 in enumerate(node_pool[idx1 + 1 :]):
                v1 = (n_pool - 2) * nd1._nj_distances[nd2]
                qvalue = v1 - nd1._nj_xsub - nd2._nj_xsub
                if qvalue < min_q:
                    min_q = qvalue
                    nodes_to_join = (nd1, nd2)

        # create the new node
        int_i = n_taxa - n_pool
        parent = int_i + n_taxa
        new_node = Node(parent)
        peel[int_i, 2] = parent

        # attach it to the tree
        peel[int_i, 0] = nodes_to_join[0].taxon
        peel[int_i, 1] = nodes_to_join[1].taxon
        node_pool.remove(nodes_to_join[0])
        node_pool.remove(nodes_to_join[1])

        # calculate the distances for the new node
        new_node._nj_distances = {}
        new_node._nj_xsub = 0.0
        for nd in node_pool:
            # actual node-to-node distances
            v1 = 0.0
            for node_to_join in nodes_to_join:
                v1 += nd._nj_distances[node_to_join]
            v3 = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            dist = 0.5 * (v1 - v3)
            new_node._nj_distances[nd] = dist
            nd._nj_distances[new_node] = dist

            # Adjust/recalculate the values needed for the Q-matrix
            # calculations
            new_node._nj_xsub += dist
            nd._nj_xsub += dist
            for node_to_join in nodes_to_join:
                nd._nj_xsub -= node_to_join._nj_distances[nd]

        # calculate the branch lengths
        if n_pool > 2:
            v1 = 0.5 * nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            v4 = (
                1.0
                / (2 * (n_pool - 2))
                * (nodes_to_join[0]._nj_xsub - nodes_to_join[1]._nj_xsub)
            )
            delta_f = v1 + v4
            delta_g = nodes_to_join[0]._nj_distances[nodes_to_join[1]] - delta_f
            blens[nodes_to_join[0].taxon] = delta_f
            blens[nodes_to_join[1].taxon] = delta_g
        else:
            dist = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
            blens[nodes_to_join[0].taxon] = dist / 2.0
            blens[nodes_to_join[1].taxon] = dist / 2.0

        # clean up
        for node_to_join in nodes_to_join:
            node_to_join._nj_distances = {}
            node_to_join._nj_xsub = 0.0

        # add the new node to the pool of nodes
        node_pool.append(new_node)

        # adjust count
        n_pool -= 1
    blens = np.maximum(blens, eps)
    return peel, blens


cpdef nj_np_old(np.ndarray[np.double_t, ndim=2] pdm):
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
        pdm = np.concatenate((pdm[:right, :], np.expand_dims(dist_p, 0), pdm[right + 1 :, :]), axis=0)
        pdm = np.concatenate(
            (pdm[:, :right], np.expand_dims(dist_p, -1), pdm[:, right + 1 :]), axis=1
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
    cdef np.int_t n_pdm = len(pdm)
    if mask is False:
        mask = np.full((n_pdm,), False)
    cdef np.int_t n_active = np.sum(~mask)

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
