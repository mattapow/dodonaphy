from heapq import heapify, heappush, heappop
from collections import deque
from . import tree, utils, poincare
from .edge import u_edge
import Cutils
import torch
import numpy as np
from collections import defaultdict


def make_peel_incentre(leaf_locs):
    return make_peel_tips(leaf_locs, method='incentre')


def make_peel_geodesic(leaf_locs):
    return make_peel_tips(leaf_locs, method='geodesic')


def make_peel_tips(leaf_locs, method='incentre'):
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

    leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
    edge_list = utils.get_pdm_tips(leaf_r, leaf_dir)

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
        # queue.sort()
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

        if method == 'geodesic':
            int_locs[int_i] = poincare.hyp_lca(from_point, to_point)
        elif method == 'incentre':
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
            if i < leaf_node_count:
                dist_ij = utils.hyperbolic_distance_locs(leaf_locs[i], int_locs[int_i])
            else:
                dist_ij = utils.hyperbolic_distance_locs(int_locs[i-leaf_node_count], int_locs[int_i])
            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = torch.log(torch.cosh(dist_ij))
            heappush(queue, u_edge(dist_ij, i, cur_internal))
        int_i += 1

    return peel, int_locs


def make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
    """Create a tree represtation (peel) from its hyperbolic embedic data

    Args:
        leaf_r (1D tensor): radius of the leaves
        leaf_dir (2D tensor): directional tensors of leaves
        int_r (1D tensor): radius of internal nodes
        int_dir (2D tensor): directional tensors of internal nodes
    """
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0] + int_r.shape[0]
    edge_list = Cutils.get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1))

    # construct a minimum spanning tree among the internal nodes
    queue = []  # queue here is a min-heap
    heapify(queue)
    visited = node_count * [False]  # visited here is a boolen list
    heappush(queue, u_edge(0, 0, 0))  # add a start_edge
    # heappush(queue, edge_list[0][0])    # add any edge from the edgelist as the start_edge
    mst_adjacencies = defaultdict(list)
    visited_count = open_slots = 0

    while queue.__len__() != 0 and visited_count < node_count:
        e = heappop(queue)

        # ensure the destination node has not been visited yet
        # internal nodes can have up to 3 adjacencies, of which at least
        # one must be internal
        # leaf nodes can only have a single edge in the MST
        is_valid = True
        if visited[e.to_]:
            is_valid = False

        if e.from_ < leaf_node_count and mst_adjacencies[e.from_].__len__() > 0:
            is_valid = False

        if e.to_ < leaf_node_count and mst_adjacencies[e.to_].__len__() > 0:
            is_valid = False

        if e.from_ >= leaf_node_count:
            if mst_adjacencies[e.from_].__len__() == 2:
                found_internal = e.to_ >= leaf_node_count
                if mst_adjacencies[e.from_][0] >= leaf_node_count:
                    found_internal = True
                if mst_adjacencies[e.from_][1] >= leaf_node_count:
                    found_internal = True
                if not found_internal and visited_count < node_count - 1:
                    is_valid = False
            elif mst_adjacencies[e.from_].__len__() == 3:
                is_valid = False

        # don't use the last open slot unless this is the last node
        if open_slots == 1 and e.to_ < leaf_node_count and visited_count < node_count - 1:
            is_valid = False
        if is_valid:
            if e.to_ is not e.from_:
                mst_adjacencies[e.from_].append(e.to_)
                mst_adjacencies[e.to_].append(e.from_)

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

    # prune internal nodes that don't create a bifurcation
    to_check = deque()  # performs better than list Re stack
    for n in range(leaf_node_count, mst_adjacencies.__len__()):
        if mst_adjacencies[n].__len__() < 3:
            to_check.append(n)

    unused = []
    while to_check.__len__() > 0:
        n = to_check.pop()
        # to_check.pop()
        if mst_adjacencies[n].__len__() == 1:
            neighbour = mst_adjacencies[n][0]
            mst_adjacencies[n].clear()
            for i in range(mst_adjacencies[neighbour].__len__()):
                if mst_adjacencies[neighbour][i] == n:
                    mst_adjacencies[neighbour].pop(
                        mst_adjacencies[neighbour][0] + i)

            unused.append(n)
            to_check.append(neighbour)
        elif mst_adjacencies[n].__len__() == 2:
            n1 = mst_adjacencies[n][0]
            n2 = mst_adjacencies[n][1]
            mst_adjacencies[n].clear()
            for i in range(mst_adjacencies[n1].__len__()):
                if mst_adjacencies[n1][i] == n:
                    mst_adjacencies[n1][i] = n2

            for i in range(mst_adjacencies[n2].__len__()):
                if mst_adjacencies[n2][i] == n:
                    mst_adjacencies[n2][i] = n1

            unused.append(n)

    # transform the MST into a binary tree.
    # find any nodes with more than three adjacencies and introduce
    # intermediate nodes to reduce the number of adjacencies
    if unused.__len__() > 0:
        for n in range(mst_adjacencies.__len__()):
            while mst_adjacencies[n].__len__() > 3:
                new_node = unused[-1]
                unused.pop(unused[-1] - 1)
                move_1 = mst_adjacencies[n][-1]
                move_2 = mst_adjacencies[n][0]
                mst_adjacencies[n].pop(mst_adjacencies[n][-1] - 1)
                mst_adjacencies[n][0] = new_node
                # link up new node
                mst_adjacencies[new_node].append(move_1)
                mst_adjacencies[new_node].append(move_2)
                mst_adjacencies[new_node].append(n)
                for move in {move_1, move_2}:
                    for i in range(mst_adjacencies[move].__len__()):
                        if mst_adjacencies[move][i] == n:
                            mst_adjacencies[move][i] = new_node

    # add a fake root above node 0: "outgroup" rooting
    zero_parent = mst_adjacencies[0][0]
    mst_adjacencies[node_count].append(0)
    mst_adjacencies[node_count].append(zero_parent)
    # mst_adjacencies.append({0, zero_parent})
    fake_root = mst_adjacencies.__len__() - 1
    mst_adjacencies[0][0] = fake_root
    for i in range(mst_adjacencies[zero_parent].__len__()):
        if mst_adjacencies[zero_parent][i] == 0:
            mst_adjacencies[zero_parent][i] = fake_root

    # make peel via post-order
    peel = []
    visited = (node_count + 1) * [False]  # all nodes + the fake root
    tree.post_order_traversal(
        mst_adjacencies, fake_root, peel, visited)

    return np.array(peel, dtype=np.intc)
