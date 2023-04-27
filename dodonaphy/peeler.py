from heapq import heapify, heappop, heappush

import numpy as np
import torch
import torch.nn.functional as F

from dodonaphy import poincare, utils, Chyp_np, Cpeeler
from dodonaphy.edge import Edge

eps = np.finfo(np.double).eps
large_value = 1e9


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
            (leaf_locs[:g, :], new_loc.unsqueeze(dim=-1).T, leaf_locs[g + 1:, :]),
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


def nj_torch(dists_leaves, tau=None):
    """Generate Neighbour joining tree with gradients.

    Args:
        dists_leaves: pairwise distance between leaves.
        tau: temperature. Set to None to use regular argmin.
             As temperature -> 0, soft_argmin becomes argmin.

    Returns:
        tuple: (peel, blens)
    """
    if tau is None:
        return Cpeeler.nj_np(dists_leaves)
    leaf_node_count = len(dists_leaves)
    node_count = 2 * leaf_node_count - 1

    # augment the distance matrix with blank internal nodes
    # fill these with a large distance
    fill_value = large_value
    right_fill = torch.full((leaf_node_count, node_count-leaf_node_count), fill_value, dtype=torch.double)
    bottom_fill = torch.full((node_count-leaf_node_count, node_count), fill_value, dtype=torch.double)
    dists = torch.hstack((dists_leaves, right_fill))
    dists = torch.vstack((dists, bottom_fill))
    dists.fill_diagonal_(0.0).requires_grad_()

    peel = np.zeros((leaf_node_count - 1, 3), dtype=int)
    blens = torch.zeros(node_count - 1, dtype=torch.double).requires_grad_()

    mask = torch.tensor(node_count * [False])
    mask[leaf_node_count:] = True

    for int_i in range(leaf_node_count - 1):
        Q = compute_Q(dists, mask)
        if tau is None:
            left, right = unravel_index(Q.argmin(), Q.shape)
            dist_p = get_new_dist(dists, mask, left, right)
            dist_pl = dist_p[left]
            dist_lr = dists[left][right]
        else:
            right_float, left_float = soft_argmin(torch.tril(Q), tau)
            hot_r = index_to_one_hot(right_float, len(Q))
            hot_l = index_to_one_hot(left_float, len(Q))
            dist_p, dist_pl, dist_lr = get_new_dist_soft(dists, mask, hot_l, hot_r)
            left, right = int(torch.round(left_float)), int(torch.round(right_float))
            dist_p[left] = dist_pl.clone()

        parent = int_i + leaf_node_count
        peel[int_i, 0] = left
        peel[int_i, 1] = right
        peel[int_i, 2] = parent

        dist_pr = torch.clamp(dist_lr - dist_pl, min=eps)
        dist_p[right] = dist_pr

        blens = blens.scatter(0, left_float.long().unsqueeze(0), dist_pl.unsqueeze(0)).clone()
        blens = blens.scatter(0, right_float.long().unsqueeze(0), dist_pr.unsqueeze(0)).clone()
        # blens[left] = dist_pl.clone()
        # blens[right] = dist_pr.clone()

        # place the new node as the next internal node in dists
        dist_p[parent] = 0.0
        dists = torch.vstack((dists[:parent, :], dist_p.unsqueeze(0), dists[parent+1:, :]))
        dists = torch.hstack((dists[:, :parent], dist_p.unsqueeze(-1), dists[:, parent+1:]))

        mask[left] = True
        mask[right] = True
        mask[parent] = False
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
    Q = (n_active - 2) * pdm - sum_i - sum_i.T

    if fill_value is None:
        fill_value = large_value
    Q.masked_fill_(mask_2d, fill_value)
    Q.fill_diagonal_(fill_value)
    Q = 0.5 * (Q + Q.T)
    return Q


def unravel_index(index, shape):
    row = torch.div(index, shape[-1], rounding_mode='trunc')
    col = torch.fmod(index, shape[-1])
    return row, col


def soft_sort(s, tau):
    s_sorted = s.sort(descending=True, dim=1)[0]
    pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
    P_hat = pairwise_distances.softmax(-1)
    return P_hat


def soft_argmin(Q, tau):
    "Soft argmin function of matrix that breaks ties with cumsum."
    assert Q.ndim == 2
    # there may be ties in Q
    Q_flat_ties = Q.view(-1)
    P_hat_ties = soft_sort(Q_flat_ties.unsqueeze(0).unsqueeze(-1), tau)
    # choose the last of any ties by multiplying by the cumulative sum
    Q_flat = P_hat_ties[:, -1] * torch.cumsum(P_hat_ties[:, -1], -1)
    P_hat = soft_sort(-Q_flat.unsqueeze(-1), tau)

    # permutation matrix to index
    unravel_indices = torch.arange(Q.numel(), dtype=Q.dtype)
    soft_indices = (P_hat[:, -1] * unravel_indices).sum((-1, 0))
    # flattened index to (row, col)
    soft_row, soft_col = unravel_index(soft_indices, Q.shape)
    return soft_row, soft_col


def index_to_one_hot(index: torch.double, size):
    one_hot = torch.zeros(size, dtype=torch.double)
    one_hot.scatter_(0, index.unsqueeze(0).long(), 1)
    return one_hot


def get_new_dist_soft(pdm, mask, hot_f, hot_g):
    dist_f = hot_f @ pdm
    dist_g = hot_g @ pdm
    dist_fg = hot_g @ dist_f
    dist_u = 0.5 * (dist_f + dist_g - dist_fg)

    n_active = sum(~mask)
    n_active = torch.clamp(n_active, min=3)

    mask_2d = ~torch.outer(~mask, ~mask)
    sum_pdm = torch.sum(pdm * ~mask_2d, dim=-1)

    dist_uf = 0.5 * dist_fg + (sum_pdm @ hot_f - sum_pdm @ hot_g) / (2 * (n_active - 2))
    dist_uf = torch.clamp(dist_uf, min=eps)
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
