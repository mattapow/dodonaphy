from heapq import heapify, heappop, heappush

import numpy as np
import torch

from . import poincare, utils, Chyp_np
from .node import Node
from .edge import Edge

eps = np.finfo(np.double).eps


def make_soft_peel_tips(leaf_locs, connector="geodesics", curvature=-torch.ones(1)):
    """Recursively generate a tree with gradients
    Args:
        leaf_locs (array): Location in of the tips in the Poincare disk

    Returns:
        tuple: (peel, int_locs)
    """
    if connector != "geodesics":
        raise NotImplementedError()
    if not torch.isclose(curvature, -torch.ones(1)):
        raise NotImplementedError()
    dims = leaf_locs.shape[1]
    leaf_node_count = leaf_locs.shape[0]
    int_node_count = leaf_locs.shape[0] - 2
    node_count = leaf_locs.shape[0] * 2 - 2

    int_locs = torch.zeros(int_node_count + 1, dims, dtype=torch.double)
    leaf_locs = leaf_locs.double()
    peel = np.zeros((int_node_count + 1, 3), dtype=int)
    blens = torch.zeros(node_count, dtype=torch.double)
    mask = torch.tensor(leaf_node_count * [False])
    node_map = torch.arange(leaf_node_count)

    for int_i in range(int_node_count + 1):
        pdm = utils.get_plca(leaf_locs.clone(), as_torch=True)
        pdm_mask = pdm * torch.outer(~mask, ~mask)
        pdm_tril = torch.tril(pdm_mask)
        local_inf = torch.max(pdm_mask) + 1.0
        pdm_nozero = torch.where(pdm_tril != 0, pdm_tril, -local_inf)
        hot_from, hot_to = soft_argmin_one_hot(-pdm_nozero, tau=1e-8)
        f = torch.where(hot_to == torch.max(hot_to))[0]
        g = torch.where(hot_from == torch.max(hot_from))[0]
        from_loc = hot_from @ leaf_locs
        to_loc = hot_to @ leaf_locs

        cur_internal = int_i + leaf_node_count
        peel[int_i, 0] = node_map[f]
        peel[int_i, 1] = node_map[g]
        peel[int_i, 2] = cur_internal

        new_loc = poincare.hyp_lca(from_loc, to_loc)
        blens[node_map[f]] = poincare.hyp_lca(leaf_locs[f], new_loc, return_coord=False)
        blens[node_map[g]] = poincare.hyp_lca(leaf_locs[g], new_loc, return_coord=False)

        # replace leaf_loc[g] by u
        # leaf_locs[g] = new_loc
        leaf_locs = torch.cat(
            (leaf_locs[:g, :], new_loc.unsqueeze(dim=-1).T, leaf_locs[g + 1 :, :]),
            dim=0,
        )
        int_locs[int_i] = new_loc
        node_map[g] = cur_internal
        mask[f] = True
    return peel, int_locs, blens


def make_hard_peel_geodesic(leaf_locs, matsumoto=False):
    """Generate a tree recursively using the closest two points.
    Curvature must be -1.0

    Args:
        leaf_locs (array): Location in of the tips in the Hyperboloid.

    Returns:
        tuple: (peel, int_locs)
    """
    dims = leaf_locs.shape[1]
    leaf_node_count = leaf_locs.shape[0]
    int_node_count = leaf_locs.shape[0] - 2
    node_count = leaf_locs.shape[0] * 2 - 2

    edge_list = utils.get_plca_np(leaf_locs, as_numpy=False)
    for node in edge_list:
        for edge in node:
            edge.distance = -edge.distance

    int_locs = np.zeros((int_node_count + 1, dims), dtype=np.double)
    peel = np.zeros((int_node_count + 1, 3), dtype=int)
    visited = node_count * [False]

    queue = []
    heapify(queue)
    for i in range(len(edge_list)):
        for j in range(i):
            heappush(queue, edge_list[i][j])

    int_i = 0
    while int_i < int_node_count + 1:
        e = heappop(queue)
        if visited[e.from_] | visited[e.to_]:
            continue

        # create a new internal node to link these
        cur_internal = int_i + leaf_node_count

        if e.from_ < leaf_node_count:
            from_point = leaf_locs[e.from_]
        else:
            from_point = int_locs[e.from_ - leaf_node_count]
        if e.to_ < leaf_node_count:
            to_point = leaf_locs[e.to_]
        else:
            to_point = int_locs[e.to_ - leaf_node_count]

        poin_from = Chyp_np.hyper_to_poincare(from_point)
        poin_to = Chyp_np.hyper_to_poincare(to_point)
        poin_lca = poincare.hyp_lca_np(poin_from, poin_to)
        int_locs[int_i] = Chyp_np.poincare_to_hyper(poin_lca)

        peel[int_i][0] = e.from_
        peel[int_i][1] = e.to_
        peel[int_i][2] = cur_internal
        visited[e.from_] = True
        visited[e.to_] = True

        # add all pairwise distances between the new node and other active nodes
        for i in range(cur_internal):
            if visited[i]:
                continue
            if i < leaf_node_count:
                poin_to = Chyp_np.hyper_to_poincare(leaf_locs[i])
                poin_from = Chyp_np.hyper_to_poincare(int_locs[int_i])
                dist_ij = -poincare.hyp_lca_np(poin_to, poin_from, return_coord=False)
            else:
                poin_to = Chyp_np.hyper_to_poincare(int_locs[i - leaf_node_count])
                poin_from = Chyp_np.hyper_to_poincare(int_locs[int_i])
                dist_ij = -poincare.hyp_lca_np(
                    poin_to,
                    poin_from,
                    return_coord=False,
                )
            if matsumoto:
                dist_ij = np.log(np.cosh(dist_ij))
            heappush(queue, Edge(dist_ij, i, cur_internal))
        int_i += 1
    return peel, int_locs


def nj_np(pdm):
    """Calculate neighbour joining tree.
    Credit to Dendropy for python implentation.

    Args:
        pdm (ndarray): Pairwise distance matrix
    """

    n_pool = len(pdm)
    n_taxa = len(pdm)
    n_ints = n_taxa - 1
    node_count = 2 * n_taxa - 2

    peel = np.zeros((n_ints, 3), dtype=int)
    blens = np.zeros(node_count, dtype=np.double)

    # initialise node pool
    node_pool = [Node(taxon=taxon) for taxon in range(n_pool)]

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
        min_q = None
        n_pool = len(node_pool)
        for idx1, nd1 in enumerate(node_pool[:-1]):
            for _, nd2 in enumerate(node_pool[idx1 + 1 :]):
                v1 = (n_pool - 2) * nd1._nj_distances[nd2]
                qvalue = v1 - nd1._nj_xsub - nd2._nj_xsub
                if min_q is None or qvalue < min_q:
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


def nj_torch(pdm, tau=None):
    """Generate Neighbour joining tree with gradients.

    Args:
        pdm: pairwise distance.
        tau: temperature. Set to None to use regular argmin.
             As temperature -> 0, soft_argmin becomes argmin.

    Returns:
        tuple: (peel, blens)
    """
    if tau is None:
        return nj_np(pdm)
    leaf_node_count = len(pdm)
    node_count = 2 * leaf_node_count - 2

    peel = np.zeros((leaf_node_count - 1, 3), dtype=int)
    blens = torch.zeros(node_count, dtype=torch.double)

    mask = torch.tensor(leaf_node_count * [False])
    node_map = torch.arange(leaf_node_count)

    for int_i in range(leaf_node_count - 1):
        Q = compute_Q(pdm, mask)
        if tau is None:
            left, right = unravel_index(Q.argmin(), Q.shape)
            dist_p = get_new_dist(pdm, mask, left, right)
            dist_pl = dist_p[left]
            dist_lr = pdm[left][right]
        else:
            hot_r, hot_l = soft_argmin_one_hot(torch.tril(Q), tau=tau)
            dist_p, dist_pl, dist_lr = get_new_dist_soft(pdm, mask, hot_l, hot_r)
            left = torch.where(hot_l == torch.max(hot_l))[0]
            right = torch.where(hot_r == torch.max(hot_r))[0]
            dist_p[left] = dist_pl
            dist_p[right] = 0

        parent = int_i + leaf_node_count
        peel[int_i, 0] = node_map[left]
        peel[int_i, 1] = node_map[right]
        peel[int_i, 2] = parent

        blens[node_map[left]] = dist_pl
        blens[node_map[right]] = torch.clamp(dist_lr - dist_pl, min=eps)

        # replace right by dist_p in the pdm
        pdm = torch.vstack((pdm[:right, :], dist_p, pdm[right + 1 :, :]))
        pdm = torch.hstack(
            (pdm[:, :right], dist_p.unsqueeze(dim=-1), pdm[:, right + 1 :])
        )
        node_map[right] = parent
        mask[left] = True

    return peel, blens


def compute_Q(pdm, mask=False, fill_value=None):
    """Compute the Q matrix for Neighbour joining.

    Args:
        pdm (ndarray): [description]
        n_active (long): [description]

    Returns:
        [type]: [description]
    """
    n_pdm = len(pdm)
    if mask is False:
        mask = torch.full((n_pdm,), False)
    n_active = sum(~mask)

    mask_2d = ~torch.outer(~mask, ~mask)
    sum_pdm = torch.sum(pdm * ~mask_2d, axis=1, keepdims=True)
    sum_i = torch.repeat_interleave(sum_pdm, n_pdm, dim=1)
    sum_j = torch.repeat_interleave(sum_pdm.T, n_pdm, dim=0)
    Q = (n_active - 2) * pdm - sum_i - sum_j

    if fill_value is None:
        fill_value = torch.finfo(torch.double).max / 2.0
    Q.masked_fill_(mask_2d, fill_value)
    Q.fill_diagonal_(fill_value)
    Q = 0.5 * (Q + Q.T)
    return Q


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="floor")
    return tuple(reversed(out))


def soft_sort(s, tau):
    s_sorted = s.sort(descending=True, dim=1)[0]
    pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
    P_hat = pairwise_distances.softmax(-1)
    return P_hat


def soft_argmin_one_hot(a_2d, tau):
    """Returns one-hot vectors indexing the minimum of a 2D tensor.
        If a_2d.requires_grad then so do the returned tensors.
        If there are multiple minimal values then the indices of the first minimal value are returned.

    Args:
        a_2d (tensor): 2D tensor
        tau (float): Temperature. See soft_sort

    Returns:
        tuple(tensor, tensor): One-hot row and column indexes of minimum of a.
    """
    a_flip = torch.flipud(a_2d)
    one_hot_i_flip = soft_argmin_row(a_flip, tau)
    one_hot_j = soft_argmin_row(one_hot_i_flip @ a_flip, tau)
    one_hot_i = torch.flip(one_hot_i_flip.unsqueeze(1), dims=(0, 1)).squeeze()
    return one_hot_i, one_hot_j


def soft_argmin_row(a, tau):
    """Return index of column (dim=0) with minimum of a. Break ties with last index."""
    many_hot_i = soft_argmin(a, tau)
    many_hot_i_notie = torch.cumsum(many_hot_i, dim=-1) * many_hot_i
    return soft_argmin(-many_hot_i_notie.unsqueeze(dim=0), tau)


def soft_argmin(a, tau):
    """Return a one hot vector indexing the column with the minumum of the input a.

    Args:
        a (tensor): A 1D or 2D (m x n) tensor, m>=1, n>1.
        tau (float): Temperature for soft_sort.

    Returns:
        tensor: One-hot vector indexing the minimum of in the dim=1 dimension.
    """
    if a.ndim == 1:
        a = a.unsqueeze(dim=0)
    a_3d = a.unsqueeze(-1)
    permute = soft_sort(a_3d, tau).squeeze()
    if a.shape[0] == 1:
        a_reshape = a_3d.squeeze(dim=0)
    else:
        a_reshape = a_3d.squeeze(dim=2)
    row_min = torch.sum(permute[:, -1] * a_reshape, -1)
    return soft_sort(row_min.unsqueeze(-1).unsqueeze(0), tau).squeeze()[-1]


def get_new_dist_soft(pdm, mask, hot_f, hot_g):
    dist_f = hot_f @ pdm
    dist_g = hot_g @ pdm
    dist_fg = hot_g @ dist_f
    dist_u = 0.5 * (dist_f + dist_g - dist_fg)

    n_active = torch.clamp(sum(~mask), min=3)
    mask_2d = ~torch.outer(~mask, ~mask)
    sum_pdm = torch.sum(pdm * ~mask_2d, dim=-1)
    dist_uf = torch.clamp(
        0.5 * dist_fg + (sum_pdm @ hot_f - sum_pdm @ hot_g) / (2 * (n_active - 2)),
        min=eps,
    )
    return dist_u, dist_uf, dist_fg


def get_new_dist(pdm, mask, f, g):
    """Get neighbour joining distances to new node."""
    # get distance all taxa to new node u
    dist_u = 0.5 * (pdm[f] + pdm[g] - pdm[f][g])

    # get distance from u to f
    n_active = torch.clamp(sum(~mask), min=3)
    mask_2d = ~torch.outer(~mask, ~mask)
    sum_pdm = torch.sum(pdm * ~mask_2d, dim=-1)
    dist_u[f] = torch.clamp(
        0.5 * pdm[f][g] + (sum_pdm[f] - sum_pdm[g]) / (2 * (n_active - 2)), min=eps
    )
    dist_u[g] = 0
    return dist_u
