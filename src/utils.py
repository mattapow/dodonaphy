import math
from heapq import heapify, heappush, heappop
from collections import deque
import numpy as np
import torch


class u_edge:
    def __init__(self, distance, node_1, node_2):
        self.distance = distance
        self.from_ = node_1
        self.to_ = node_2


class utilFunc:
    def __init__(self):
        pass

    @staticmethod
    def hyperbolic_distance(r1, r2, directional1, directional2, curvature):
        """Generates hyperbolic distance between two points in poincoire ball

        Args:
            r1 (tensor): radius of point 1
            r2 (tensor): radius of point 2
            directional1 (tensor): directional of point 1
            directional2 ([type]): directional of point 2
            curvature (integer): curvature

        Returns:
            tensor: distance between point 1 and point 2
        """
        dpl = torch.empty(2)
        dpl[0] = torch.dot(directional1, directional2)
        dpl[1] = torch.tensor([-1.0], requires_grad=True)
        iprod = dpl[0]
        dpl[0] = 2 * (torch.pow(r1, 2) + torch.pow(r2, 2) - 2*r1 *
                      r2*iprod)/((1-torch.pow(r1, 2) * (1-torch.pow(r2, 2))))
        dpl[1] = 0.0
        acosharg = 1.0 + max(dpl)
        # hyperbolic distance between points i and j
        dist = 1/math.sqrt(curvature) * torch.cosh(acosharg)
        return dist + 0.000000000001  # add a tiny amount to avoid zero-length branches

    @staticmethod
    def make_peel(leaf_r, leaf_dir, int_r, int_dir, location_map):
        """Create a tree represtation (peel) from its hyperbolic embedic data

        Args:
            leaf_r (1D tensor): radius of the leaves
            leaf_dir (2D tensor): directional tensors of leaves 
            int_r (1D tensor): radius of internal nodes
            int_dir (2D tensor): directional tensors of internal nodes
            location_map (numpy array): node location map
        """
        leaf_node_count = leaf_r.shape[0]
        node_count = leaf_r.shape[0] + int_r.shape[0]
        edge_list = node_count*[[]]
        # edge_list = np.array(edge_list, dtype=u_edge)

        for i in range(node_count):
            for j in range(max(i+1, leaf_node_count), node_count):
                dist_ij = 0

                if(i < leaf_node_count):
                    dist_ij = self.hyperbolic_distance(self,
                                                       leaf_r[i],
                                                       leaf_dir[i],
                                                       int_r[j-leaf_node_count],
                                                       int_dir[j -
                                                               leaf_node_count],
                                                       1.0)
                else:
                    i_node = i - leaf_node_count
                    dist_ij = self.hyperbolic_distance(self,
                                                           int_r[i_node],
                                                           int_dir[i_node],
                                                           int_r[j-leaf_node_count],
                                                           int_dir[j -
                                                                   leaf_node_count],
                                                           1.0)

                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))

                # use negative of distance so that least dist has largest 
                # value in the priority queue
                edge_list[i].append(u_edge(dist_ij, i, j))
                edge_list[j].append(u_edge(dist_ij, j, i))

        # construct a minimum spanning tree among the internal nodes
        queue = []  # queue here is a min-heap
        heapify(queue)
        visited = node_count*[False]  # visited here is a boolen list
        start_edge = u_edge(0, 0, 0)
        heappush(queue, start_edge)
        mst_adjacencies = node_count*[[]]
        visited_count = 0
        open_slots = 0
        while queue.__len__() != 0 and visited_count < node_count:
            e = u_edge(heappop(queue))

            # ensure the destination node has not been visited yet
            # internal nodes can have up to 3 adjacencies, of which at least 
            # one must be internal
            # leaf nodes can only have a single edge in the MST
            is_valid = True
            if visited[e.to_]: is_valid = True 
            
            if e.from_ < leaf_node_count and mst_adjacencies[e.to_].__len__ > 0: is_valid = False 

            if e.from_ >= leaf_node_count:
                if mst_adjacencies[e.from_].__len__() == 2:
                    found_internal = e.to_ >= leaf_node_count
                    if mst_adjacencies[e.from_][0] >= leaf_node_count: found_internal = True 
                    if mst_adjacencies[e.from_][1] >= leaf_node_count: found_internal = True 
                    if not found_internal: is_valid = False 
                elif mst_adjacencies[e.from_].__len__() == 3:
                    is_valid = False

            # don't use the last open slot unless this is the last node
            if open_slots == 1 and e.to_ < leaf_node_count and visited_count < node_count - 1: is_valid = False 
            if is_valid:
                if e.to_ is not e.from_:
                    mst_adjacencies[e.from_].append(e.to_)
                    mst_adjacencies[e.to_].append(e.from_)

                # a new internal node has room for 2 more adjacencies
                if e.to_ != leaf_node_count: open_slots += 2 
                if e.from_ >= leaf_node_count: open_slots -= 1 

                visited[e.to_] = True
                visited_count += 1
                for new_e in edge_list[e.to_]:
                    if visited[new_e.to_]:
                        continue
                    heapq.heappush(queue, new_e)

        # prune internal nodes that don't create a bifurcation
        to_check = deque()
        for n in range(leaf_node_count, mst_adjacencies.__len__()):
            if mst_adjacencies[n].__len__() < 3: to_check.append(n) 

        unused = []
        while to_check.__len__() > 0:
            n = to_check.pop()
            to_check.pop()
            if mst_adjacencies[n].__len__() == 1:
                neighbour = mst_adjacencies[n][0]
                mst_adjacencies[n].clear()
                for i in range(mst_adjacencies[neighbour].__len__()):
                    if mst_adjacencies[neighbour][i] == n: mst_adjacencies[neighbour].pop(mst_adjacencies[neighbour][0] + i) 

                unused.append(n)
                to_check.append(neighbour)
            elif mst_adjacencies[n].__len__() == 2:
                n1 = mst_adjacencies[n][0]
                n2 = mst_adjacencies[n][1]
                mst_adjacencies[n].clear()
                for i in range(mst_adjacencies[n1].__len__()):
                    if mst_adjacencies[n1][i] == n: mst_adjacencies[n1][i] = n2 

                for i in range(mst_adjacencies[n2].__len__()):
                    if mst_adjacencies[n2][i] == n: mst_adjacencies[n2][i] = n1 

                unused.append(n)

        # initialize location_map with every node pointing to itself
        for i in range(location_map.__len__()):
            location_map[i] = i

        # transform the MST into a binary tree.
        # find any nodes with more than three adjacencies and introduce
        # intermediate nodes to reduce the number of adjacencies
        if unused.__len__() > 0:
            for n in range(mst_adjacencies.__len__)():
                while mst_adjacencies[n].__len__() > 3:
                    new_node = unused[-1]
                    unused.pop(unused[-1]-1)
                    move_1 = mst_adjacencies[n][-1]
                    move_2 = mst_adjacencies[n][0]
                    mst_adjacencies[n].pop(mst_adjacencies[n][-1]-1)
                    mst_adjacencies[n][0] = new_node
                    # link up new node
                    mst_adjacencies[new_node].append(move_1)
                    mst_adjacencies[new_node].append(move_2)
                    mst_adjacencies[new_node].append(n)
                    for move in {move_1, move_2}:
                        for i in range(mst_adjacencies[move].__len__()):
                            if mst_adjacencies[move][i] == n: mst_adjacencies[move][i] = new_node 

                    # map the location for the new node to the original node
                    location_map[new_node] = n

        # update the location map - handles multiple reassignments
        for i in range(location_map.__len__()):
            parent = location_map[i]
            while parent != location_map[parent]:
                location_map[parent] = location_map[location_map[parent]]
                parent = location_map[parent]
            location_map[i] = parent

        # add a fake root above node 0
        zero_parent = mst_adjacencies[0][0]
        mst_adjacencies.append({0, zero_parent})
        mst_adjacencies[0][0] = mst_adjacencies.__len__() - 1
        for i in range(mst_adjacencies[zero_parent].__len__()):
            if mst_adjacencies[zero_parent][i] == 0: mst_adjacencies[zero_parent][i] = mst_adjacencies.__len__() - 1 
        location_map[mst_adjacencies.__len__() - 1] = zero_parent

        # make peel via pre-order traversal
        visited = [node_count]
        node_stack = []
        node_stack.append(mst_adjacencies[0][0])
        peelI = peel.size()
        while node_stack.__len__() != 0:
            cur_node = node_stack.pop()
            if mst_adjacencies[cur_node].__len__() < 2:
                continue    # leaf node, nothing to do
            # remove already-visited nodes from adjacencies, leaving just two children
            for iter in mst_adjacencies[cur_node]:
                if visited[iter]:
                    mst_adjacencies[cur_node].remove(iter)
            # peel entries are child, child, parent
            # cur_node should always have two adjacencies
            peelI = peelI - 1
            peel[peelI] = {mst_adjacencies[cur_node][0],
                           mst_adjacencies[cur_node][1], cur_node}
            node_stack.append(peel[peelI][0])
            node_stack.append(peel[peelI][1])
            visited[cur_node] = True
        for i in range(peel.__len__()):
            for j in range(peel[i].__len__()):
                peel[i][j] += 1
        for i in range(location_map.__len__()):
            location_map[i] += 1

    # def compute_LL(S, L, bcount, D, tipdata, leaf_r, leaf_dir, int_r, int_dir):
    #     partials =

    # def compute_branch_lengths(S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir):
    #     bcount = 2*S-2
    #     blens = bcount*[]
    #     for b in range(1, S-1):
    #         directional1 = D*[]
    #         directional2 = D*[]
    #         r2 = int_r[location_map[peel[b, 3]]-S]
    #         directional2 = int_dir[location_map[peel[b, 3]]-S, ]
