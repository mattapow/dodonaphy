import math
from heapq import heapify, heappush, heappop
from collections import deque

class u_edge:
    def __init__(self, distance, node_1, node_2):
        self.distance = distance
        self.from_ = node_1
        self.to_ = node_2

class utilFunc:
    def __init__(self):
        self.x = 0

    # hyperbolic distance function, translated to C++ from the R hydra package
    # def hyperbolic_distance(self, loc_r1, loc1, loc_r2, loc2, curvature):
    #     # compute the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
    #     prodsum = 0
    #     for i in range(loc1.size):
    #         prodsum += (loc1[i] + loc2[i])

    #     # force between numerical -1.0 and 1.0 to eliminate rounding errors
    #     iprod = -1.0 if prodsum < -1.0 else prodsum
    #     iprod = 1.0 if iprod > 1.0 else iprod

    #     # hyperbolic 'angle'; force numerical >= 1.0
    #     r1 = loc_r1
    #     r2 = loc_r2
    #     hyper_angle = 2.0*(pow(r1, 2) + pow(r2, 2) - 2.0 *
    #                        r1*r2*iprod)/((1 - pow(r1, 2))*(1 - pow(r2, 2)))
    #     print(hyper_angle)
    #     hyper_angle = 1.0 + (0 if hyper_angle < 0 else hyper_angle)

    #     # final hyperbolic distance between points i and j
    #     distance = 1/math.sqrt(curvature) * math.acosh(hyper_angle)
    #     return distance

    def make_peel(self, leaf_r, leaf_dir, int_r, int_dir, location_map):
        leaf_node_count = leaf_r.size
        node_count = leaf_r.size + int_r.size
        edge_list = node_count*[[]]

        for i in range(node_count):
            for j in range(max(i+1, leaf_node_count), node_count):
                dist_ij = 0

                if(i < leaf_node_count):
                    dist_ij = utilFunc.hyperbolic_distance(self,
                                                           leaf_r[i], leaf_dir[i], int_r[j-leaf_node_count], int_dir[j-leaf_node_count], 1.0)
                else:
                    i_node = i - leaf_node_count
                    dist_ij = utilFunc.hyperbolic_distance(self,
                                                           int_r[i_node], int_dir[i_node], int_r[j-leaf_node_count], int_dir[j-leaf_node_count], 1.0)

                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = math.log(math.cosh(dist_ij))

                # // use negative of distance so that least dist has largest value in the priority queue
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
            e = heappop(queue)

            # ensure the destination node has not been visited yet
            # internal nodes can have up to 3 adjacencies, of which at least one must be internal
            # leaf nodes can only have a single edge in the MST
            is_valid = True
            is_valid = True if visited[e.to_]
            is_valid = False if e.from_ < leaf_node_count and mst_adjacencies[e.to_].__len__ > 0

            if e.from_ >= leaf_node_count:
                if mst_adjacencies[e.from_].__len__() == 2:
                    found_internal = e.to_ >= leaf_node_count
                    found_internal = True if mst_adjacencies[e.from_][0] >= leaf_node_count
                    found_internal = True if mst_adjacencies[e.from_][1] >= leaf_node_count
                    is_valid = False if not found_internal
                elif mst_adjacencies[e.from_].__len__() == 3:
                    is_valid = False

            # don't use the last open slot unless this is the last node
            is_valid = False if open_slots == 1 and e.to_ < leaf_node_count and visited_count < node_count - 1
            if is_valid:
                if e.to_ is not e.from_:
                    mst_adjacencies[e.from_].append(e.to_)
                    mst_adjacencies[e.to_].append(e.from_)

                # a new internal node has room for 2 more adjacencies
                open_slots += 2 if e.to_ != leaf_node_count
                open_slots -= 1 if e.from_ >= leaf_node_count

                visited[e.to_] = True
                visited_count += 1
                for new_e in edge_list[e.to_]:
                    if visited[new_e.to_]:
                        continue
                    heapq.heappush(queue, new_e)

        # prune internal nodes that don't create a bifurcation
        to_check = deque()
        for n in range(leaf_node_count, mst_adjacencies.__len__()):
            to_check.append(n) if mst_adjacencies[n].__len__() < 3

        unused = []
        while to_check.__len__() > 0:
            n = to_check.pop()
            to_check.pop()
            if mst_adjacencies[n].__len__() == 1:
                neighbour = mst_adjacencies[n][0]
                mst_adjacencies[n].clear()
                for i in range(mst_adjacencies[neighbour].__len__()):
                    mst_adjacencies[neighbour].pop(mst_adjacencies[neighbour][0] + i) if mst_adjacencies[neighbour][i] == n

                unused.append(n)
                to_check.append(neighbour)
            elif mst_adjacencies[n].__len__() == 2:
                n1 = mst_adjacencies[n][0]
                n2 = mst_adjacencies[n][1]
                mst_adjacencies[n].clear()
                for i in range(mst_adjacencies[n1].__len__()):
                    mst_adjacencies[n1][i] = n2 if mst_adjacencies[n1][i] == n

                for i in range(mst_adjacencies[n2].__len__()):
                    mst_adjacencies[n2][i] = n1 if mst_adjacencies[n2][i] == n:

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
                            mst_adjacencies[move][i] = new_node if mst_adjacencies[move][i] == n

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
        mst_adjacencies.append({0,zero_parent})
        mst_adjacencies[0][0] = mst_adjacencies.__len__() - 1
        for i range(mst_adjacencies[zero_parent].__len__()):
            mst_adjacencies[zero_parent][i] = mst_adjacencies.__len__() - 1 if mst_adjacencies[zero_parent][i] == 0
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
            peel[peelI] = {mst_adjacencies[cur_node][0],mst_adjacencies[cur_node][1], cur_node}
            node_stack.append(peel[peelI][0])
            node_stack.append(peel[peelI][1])
            visited[cur_node] = True
        for i in range(peel.__len__()):
            for j in range(peel[i].__len__()):
                peel[i][j] += 1
        for i in range(location_map.__len__()):
            location_map[i] += 1
    
    def compute_LL(S,L,bcount,D,tipdata, leaf_r, leaf_dir, int_r, int_dir):
        partials =

    def compute_branch_lengths(S,D,peel,location_map, leaf_r, leaf_dir, int_r, int_dir):
        bcount = 2*S-2
        blens = bcount*[]
        for b in range(1,S-1):
            directional1 = D*[]
            directional2 = D*[]
            r2 = int_r[location_map[peel[b,3]]-S]
            directional2 = int_dir[location_map[peel[b,3]]-S,]


