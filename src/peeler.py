import math
from collections import defaultdict, deque
from heapq import heapify, heappop, heappush

import Cutils
import numpy as np
import torch

from . import poincare, tree, utils
from .edge import u_edge


def make_peel_incentre(leaf_locs, curvature=-torch.ones(1)):
    return make_peel_tips(leaf_locs, connect_method='incentre', curvature=curvature)


def make_peel_geodesic(leaf_locs):
    return make_peel_tips(leaf_locs, connect_method='geodesics')


def make_peel_tips(leaf_locs, connect_method='geodesics', curvature=-torch.ones(1)):
    """Generate a tree recursively using the incentre of the closest two points.

    Args:
        leaf_locs (array): Location in of the tips in the Poincare disk

    Returns:
        tuple: (peel, int_locs)
    """
    dims = leaf_locs.shape[1]
    leaf_node_count = leaf_locs.shape[0]
    int_node_count = leaf_locs.shape[0] - 2
    node_count = leaf_locs.shape[0] * 2 - 2

    if connect_method == 'geodesics':
        edge_list = utils.get_plca(leaf_locs)
    elif connect_method == 'incentre':
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
        edge_list = utils.get_pdm_tips(leaf_r, leaf_dir, curvature=curvature)
    else:
        raise ValueError('connect_method must be geodesics or incentre')

    int_locs = torch.zeros(int_node_count+1, dims, dtype=torch.double)
    leaf_locs = leaf_locs.double()
    peel = np.zeros((int_node_count+1, 3), dtype=np.int16)
    visited = node_count * [False]

    # queue = [edges for neighbours in edge_list for edges in neighbours]
    queue = []
    heapify(queue)
    for i in range(len(edge_list)):
        for j in range(i):
            heappush(queue, edge_list[i][j])

    int_i = 0
    while int_i < int_node_count+1:
        e = heappop(queue)
        if(visited[e.from_] | visited[e.to_]):
            continue

        # create a new internal node to link these
        cur_internal = int_i + leaf_node_count

        if e.from_ < leaf_node_count:
            from_point = leaf_locs[e.from_]
        else:
            from_point = int_locs[e.from_-leaf_node_count]
        if e.to_ < leaf_node_count:
            to_point = leaf_locs[e.to_]
        else:
            to_point = int_locs[e.to_-leaf_node_count]

        if connect_method == 'geodesics':
            int_locs[int_i] = poincare.hyp_lca(from_point, to_point)
        elif connect_method == 'incentre':
            int_locs[int_i] = poincare.incentre(from_point, to_point)

        peel[int_i][0] = e.from_
        peel[int_i][1] = e.to_
        peel[int_i][2] = cur_internal
        visited[e.from_] = True
        visited[e.to_] = True

        # add all pairwise distances between the new node and other active nodes
        for i in range(cur_internal):
            if visited[i]:
                continue
            if connect_method == 'geodesics':
                if i < leaf_node_count:
                    dist_ij = - poincare.hyp_lca(leaf_locs[i], int_locs[int_i], return_coord=False)
                else:
                    dist_ij = - poincare.hyp_lca(int_locs[i-leaf_node_count], int_locs[int_i], return_coord=False)
            elif connect_method == 'incentre':
                if i < leaf_node_count:
                    dist_ij = Cutils.hyperbolic_distance_lorentz(leaf_locs[i], int_locs[int_i])
                else:
                    dist_ij = Cutils.hyperbolic_distance_lorentz(int_locs[i-leaf_node_count], int_locs[int_i])
                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))
            heappush(queue, u_edge(dist_ij, i, cur_internal))
        int_i += 1

    return peel, int_locs


def make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1), start_node=None):
    """Create a tree represtation (peel) from its hyperbolic embedic data

    Args:
        leaf_r (1D tensor): radius of the leaves
        leaf_dir (2D tensor): directional tensors of leaves
        int_r (1D tensor): radius of internal nodes
        int_dir (2D tensor): directional tensors of internal nodes
    """
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0] + int_r.shape[0]
    edge_list = Cutils.get_pdm_np(leaf_r, leaf_dir, int_r, int_dir, curvature=curvature, dtype='dict')

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
            e.to_, e.from_, adjacency, visited, leaf_node_count, node_count, open_slots, visited_count)

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

    return np.array(peel, dtype=np.intc)


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
                raise RuntimeError('No candidates found')
    return start_edge


def get_smallest_edge(edge_list, node_count, leaf_node_count):
    """
    Find the shortest edge between a leaf and internal node.
    """
    start_edge = u_edge(np.inf, -1, -1)
    int_node_count = node_count - leaf_node_count
    for i in range(int_node_count):
        int_i = i + leaf_node_count
        for edge in edge_list[int_i]:
            isLeafInt = False
            if edge.to_ >= leaf_node_count and edge.from_ < leaf_node_count:
                isLeafInt = True
            if edge.to_ < leaf_node_count and edge.from_ >= leaf_node_count:
                isLeafInt = True

            if isLeafInt and edge < start_edge:
                if edge.to_ < edge.from_:
                    # reverse first edge so to_ is internal
                    edge = u_edge(edge.distance, edge.to_, edge.from_)
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
    leaf_node_count = int(len(adjacency)/2 + 1)
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
                    adjacency[neighbour].pop(
                        adjacency[neighbour][0] + i)

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


def is_valid_edge(to_, from_, adjacency, visited, S, node_count, open_slots, visited_count):
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


def nj(pdm):
    """Generate Neighbour joining tree.

    Args:
        leaf_r ([type]): [description]
        leaf_dir ([type]): [description]
        curvature ([type], optional): [description]. Defaults to torch.ones(1).

    Returns:
        tuple: (peel, blens)
    """
    leaf_node_count = len(pdm)
    int_node_count = leaf_node_count - 2
    node_count = leaf_node_count + int_node_count
    eps = torch.finfo(torch.double).eps

    # initialise peel and branch lengths
    peel = np.zeros((int_node_count + 1, 3), dtype=int)
    blens = torch.zeros(node_count, dtype=torch.float64)

    # construct Q matrix
    Q = compute_Q(pdm)

    # Add a mask to the Q matrix
    mask = torch.tensor(leaf_node_count * [False])

    # for each internal node
    for int_i in range(int_node_count + 1):
        # find minimum of Q matrix
        Q.masked_fill_(~torch.outer(~mask, ~mask), math.inf)
        g, f = unravel_index(Q.argmin(), Q.shape)
        # TODO: use view if unravel is problem.
        # TODO: use softsort.

        # add this branch to peel
        u = int_i + leaf_node_count
        peel[int_i, 0] = f
        peel[int_i, 1] = g
        peel[int_i, 2] = u

        # get distances to new node u
        pdm = add_pdm_node_nj(pdm, f, g)

        # add edges above f and g to blens
        blens[f] = torch.clamp(pdm[f, u], min=eps)
        blens[g] = torch.clamp(pdm[g, u], min=eps)

        # update Q matrix
        mask[f] = mask[g] = torch.tensor(True)
        mask = torch.hstack((mask, torch.tensor(False)))
        Q = update_Q(Q, pdm, mask)

    return peel, blens


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def compute_Q(pdm):
    """Compute the Q matrix for Neighbour joining.

    Args:
        pdm (ndarray): [description]

    Returns:
        [type]: [description]
    """
    n = len(pdm)
    sum_pdm = torch.sum(pdm, axis=1, keepdims=True)
    sum_i = torch.repeat_interleave(sum_pdm, n, dim=1)
    sum_j = torch.repeat_interleave(sum_pdm.T, n, dim=0)
    Q = (n - 2) * pdm - sum_i - sum_j
    Q.fill_diagonal_(0)
    return Q


def update_Q(Q, pdm_updated, mask):
    """ add new node to masked Q matrix

    Args:
        Q ([type]): [description]
        pdm_updated ([type]): [description]
        mask([type]): [description]

    Returns:
        [type]: [description]
    """
    n = len(Q)

    # new column Q_u for new node u
    sum_pdm = torch.sum(pdm_updated, dim=-1)
    Q_u = (n - 2) * pdm_updated[:-1, -1] - sum_pdm[:-1] - sum_pdm[-1]

    # add to Q matrix
    Q = torch.vstack((Q, Q_u.unsqueeze(dim=0)))
    Q_u = torch.cat((Q_u, torch.zeros(1)))
    Q = torch.hstack((Q, Q_u.unsqueeze(dim=1)))

    # update mask
    mask = ~torch.outer(~mask, ~mask)

    return Q


def add_pdm_node_nj(pdm, f, g):
    """Add a node to the pdm based on neighbour joining.

    Args:
        pdm ([type]): pair-wise distance matrix
        f ([type]): f-g are index of lowest in Q matrix
        g ([type]): f-g are index of lowest in Q matrix
    """
    # distance to new node
    n = len(pdm)

    # get distance other taxa to new node
    pdm_u = .5 * (pdm[f] + pdm[g] - pdm[f][g])

    # get distance fu and fg
    sum_pdm = torch.sum(pdm, dim=-1)
    pdm_u[f] = .5 * pdm[f][g] + (sum_pdm[f] - sum_pdm[g]) / (2 * (n - 2))
    pdm_u[g] = pdm[f][g] - pdm_u[f]

    # add new node to pdm
    pdm = torch.vstack((pdm, pdm_u.unsqueeze(dim=0)))
    pdm_u = torch.cat((pdm_u, torch.zeros(1)))
    pdm = torch.hstack((pdm, pdm_u.unsqueeze(dim=1)))

    return pdm
