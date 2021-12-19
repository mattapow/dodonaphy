from collections import defaultdict, deque
from heapq import heapify, heappop, heappush

import numpy as np
import torch

from . import poincare, tree, utils, Cpeeler, Chyperboloid_np
from .edge import Edge


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


def make_hard_peel_geodesic(leaf_locs):
    """Generate a tree recursively using th closest two points.
    Curvature must be -1.0

    Args:
        leaf_locs (array): Location in of the tips in the Poincare disk

    Returns:
        tuple: (peel, int_locs)
    """
    dims = leaf_locs.shape[1]
    leaf_node_count = leaf_locs.shape[0]
    int_node_count = leaf_locs.shape[0] - 2
    node_count = leaf_locs.shape[0] * 2 - 2

    print(type(leaf_locs))
    edge_list = utils.get_plca_np(leaf_locs, as_numpy=False)
    print(type(edge_list))
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

        int_locs[int_i] = poincare.hyp_lca_np(from_point, to_point)

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
                dist_ij = -poincare.hyp_lca_np(
                    leaf_locs[i], int_locs[int_i], return_coord=False
                )
            else:
                dist_ij = -poincare.hyp_lca_np(
                    int_locs[i - leaf_node_count],
                    int_locs[int_i],
                    return_coord=False,
                )
            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = np.log(np.cosh(dist_ij))
            heappush(queue, Edge(dist_ij, i, cur_internal))
        int_i += 1
    return peel, int_locs


def make_peel_mst(
    leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1), start_node=None
):
    """Create a tree represtation (peel) from its hyperbolic embedic data

    Args:
        leaf_r (1D tensor): radius of the leaves
        leaf_dir (2D tensor): directional tensors of leaves
        int_r (1D tensor): radius of internal nodes
        int_dir (2D tensor): directional tensors of internal nodes
    """
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0] + int_r.shape[0]
    if isinstance(leaf_r, torch.Tensor):
        leaf_r = leaf_r.detach().numpy().astype(np.double),
        leaf_dir = leaf_dir.detach().numpy().astype(np.double),
        int_r = int_r.detach().numpy().astype(np.double),
        int_dir = int_dir.detach().numpy().astype(np.double),
    edge_list = Chyperboloid_np.get_pdm(
        leaf_r,
        leaf_dir,
        int_r,
        int_dir,
        curvature=curvature,
        dtype="dict",
    )

    # queue here is a min-heap
    queue = []
    heapify(queue)

    start_edge = get_start_edge(start_node, edge_list, node_count, leaf_node_count)
    heappush(queue, start_edge)

    adjacency = defaultdict(list)
    visited_count = open_slots = 0
    visited = node_count * [False]  # visited here is a boolen list

    while queue.__len__() != 0 and visited_count < node_count:
        e = heappop(queue)

        # check if edge is valid for binary tree
        is_valid = is_valid_edge(
            e.to_,
            e.from_,
            adjacency,
            visited,
            leaf_node_count,
            node_count,
            open_slots,
            visited_count,
        )

        if is_valid:
            adjacency[e.from_].append(e.to_)
            adjacency[e.to_].append(e.from_)

            # a new internal node has room for 2 more adjacencies
            if e.to_ >= leaf_node_count:
                open_slots += 2
            if e.from_ >= leaf_node_count:
                open_slots -= 1

            visited[e.to_] = True
            visited_count += 1
            for new_e in edge_list[e.to_]:
                if visited[new_e.to_]:
                    continue
                heappush(queue, new_e)

    # transform the MST into a binary tree.
    # find any nodes with more than three adjacencies and introduce
    # intermediate nodes to reduce the number of adjacencies
    # NB: this is doing nothing
    adjacency = force_binary(adjacency)

    # add a fake root above node 0: "outgroup" rooting
    zero_parent = adjacency[0][0]
    adjacency[node_count].append(0)
    adjacency[node_count].append(zero_parent)

    fake_root = adjacency.__len__() - 1
    adjacency[0][0] = fake_root
    for i in range(adjacency[zero_parent].__len__()):
        if adjacency[zero_parent][i] == 0:
            adjacency[zero_parent][i] = fake_root

    # make peel via post-order
    peel = []
    visited = (node_count + 1) * [False]  # all nodes + the fake root
    tree.post_order_traversal(adjacency, fake_root, peel, visited)

    return np.array(peel, dtype=int)


def get_start_edge(start_node, edge_list, node_count, leaf_node_count):
    if start_node is None:
        # Use closest leaf to internal as start edge
        start_edge = get_smallest_edge(edge_list, node_count, leaf_node_count)
    elif isinstance(start_node, int):
        candidate_edges = []
        heapify(candidate_edges)
        for i in range(len(edge_list[start_node])):
            heappush(candidate_edges, edge_list[start_node][i])
        is_valid = False
        while not is_valid:
            start_edge = heappop(candidate_edges)
            if start_edge.to_ < leaf_node_count and start_edge.from_ >= leaf_node_count:
                is_valid = True
            if start_edge.from_ < leaf_node_count and start_edge.to_ >= leaf_node_count:
                is_valid = True
            if len(candidate_edges) == 0:
                raise RuntimeError("No candidates found")
    return start_edge


def get_smallest_edge(edge_list, node_count, leaf_node_count):
    """
    Find the shortest edge between a leaf and internal node.
    """
    start_edge = Edge(np.inf, -1, -1)
    int_node_count = node_count - leaf_node_count
    for i in range(int_node_count):
        int_i = i + leaf_node_count
        for edge in edge_list[int_i]:
            is_leaf_int = False
            if edge.to_ >= leaf_node_count and edge.from_ < leaf_node_count:
                is_leaf_int = True
            if edge.to_ < leaf_node_count and edge.from_ >= leaf_node_count:
                is_leaf_int = True

            if is_leaf_int and edge < start_edge:
                if edge.to_ < edge.from_:
                    # reverse first edge so to_ is internal
                    edge = Edge(edge.distance, edge.to_, edge.from_)
                start_edge = edge
    return start_edge


def force_binary(adjacency):
    """transform the MST into a binary tree.
    find any nodes with more than three adjacencies and introduce
    intermediate nodes to reduce the number of adjacencies

    Args:
        adjacency ([type]): [description]
    """
    # prune internal nodes that don't create a bifurcation
    leaf_node_count = int(len(adjacency) / 2 + 1)
    unused, adjacency = prune(leaf_node_count, adjacency)

    if unused.__len__() > 0:
        for n in range(adjacency.__len__()):
            while adjacency[n].__len__() > 3:
                new_node = unused[-1]
                unused.pop(unused[-1] - 1)
                move_1 = adjacency[n][-1]
                move_2 = adjacency[n][0]
                adjacency[n].pop(adjacency[n][-1] - 1)
                adjacency[n][0] = new_node
                # link up new node
                adjacency[new_node].append(move_1)
                adjacency[new_node].append(move_2)
                adjacency[new_node].append(n)
                for move in {move_1, move_2}:
                    for i in range(adjacency[move].__len__()):
                        if adjacency[move][i] == n:
                            adjacency[move][i] = new_node
    return adjacency


def prune(S, adjacency):
    # prune internal nodes that don't create a bifurcation
    to_check = deque()  # performs better than list Re stack
    for n in range(S, adjacency.__len__()):
        if adjacency[n].__len__() < 3:
            to_check.append(n)

    unused = []
    while to_check.__len__() > 0:
        n = to_check.pop()
        if adjacency[n].__len__() == 1:
            neighbour = adjacency[n][0]
            adjacency[n].clear()
            for i in range(adjacency[neighbour].__len__()):
                if adjacency[neighbour][i] == n:
                    adjacency[neighbour].pop(adjacency[neighbour][0] + i)

            unused.append(n)
            to_check.append(neighbour)
        elif adjacency[n].__len__() == 2:
            n1 = adjacency[n][0]
            n2 = adjacency[n][1]
            adjacency[n].clear()
            for i in range(adjacency[n1].__len__()):
                if adjacency[n1][i] == n:
                    adjacency[n1][i] = n2

            for i in range(adjacency[n2].__len__()):
                if adjacency[n2][i] == n:
                    adjacency[n2][i] = n1

            unused.append(n)
    return unused, adjacency


def is_valid_edge(
    to_, from_, adjacency, visited, S, node_count, open_slots, visited_count
):
    # ensure the destination node has not been visited yet
    # internal nodes can have up to 3 adjacencies, of which at least
    # one must be internal
    # leaf nodes can only have a single edge in the MST
    if to_ is from_:
        return False

    if visited[to_]:
        return False

    if isinstance(adjacency, dict):
        n_from = adjacency[from_].__len__()
        n_to = adjacency[to_].__len__()
        if n_from == 2:
            from_edge0 = adjacency[from_][0]
            from_edge1 = adjacency[from_][1]
    elif isinstance(adjacency, np.ndarray):
        n_from = np.sum(adjacency[from_])
        n_to = np.sum(adjacency[to_])
        if n_from == 2:
            from_edge = np.where(adjacency[from_])[0]
            from_edge0 = from_edge[0]
            from_edge1 = from_edge[1]

    if from_ < S and n_from > 0:
        return False

    if to_ < S and n_to > 0:
        return False

    is_valid = True
    if from_ >= S:
        if n_from == 2:
            found_internal = to_ >= S
            if from_edge0 >= S:
                found_internal = True
            if from_edge1 >= S:
                found_internal = True
            if not found_internal and visited_count < node_count - 2:
                is_valid = False
        elif n_from == 3:
            is_valid = False

    # don't use the last open slot unless this is the last node
    if open_slots == 1 and to_ < S and visited_count < node_count - 2:
        is_valid = False
    return is_valid


def nj(pdm, tau=None):
    """Generate Neighbour joining tree.

    Args:
        pdm: pairwise distance.
        tau: temperature. Set to None to use regular argmin.
             As temperature -> 0, soft_argmin becomes argmin.

    Returns:
        tuple: (peel, blens)
    """
    if tau is None:
        return Cpeeler.nj_np(pdm)
    leaf_node_count = len(pdm)
    node_count = 2 * leaf_node_count - 2
    eps = torch.finfo(torch.double).eps

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
        min=torch.finfo(torch.double).eps,
    )
    return dist_u, dist_uf, dist_fg


def get_new_dist(pdm, mask, f, g):
    """Get neighbour joining distances to new node."""
    eps = torch.finfo(torch.double).eps
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
