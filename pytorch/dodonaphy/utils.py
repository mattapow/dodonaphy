import math
from heapq import heapify, heappush, heappop
from collections import deque
import numpy as np
import torch
import warnings
import sys
from collections import defaultdict


class u_edge:
    def __init__(self, distance, node_1, node_2):
        self.distance = distance
        self.from_ = node_1
        self.to_ = node_2

    def __lt__(self, other):
        return self.distance < other.distance

class utilFunc:
    def __init__(self):
        pass

    @staticmethod
    def hydra(D, dim=2, curvature=1, alpha=1.1, equi_adj=0.5, **kwargs):
        """Strain minimised hyperbolic embedding
        Python Implementation of Martin Keller-Ressel's 2019 CRAN function
        hydra
        https://arxiv.org/abs/1903.08977

        Parameters
        ----------
        D : ndarray
            Pairwise distance matrix.
        dim : Int, optional
            Embedding dimension. The default is 2.
        curvature : Float, optional
            Embedding curvature. The default is 1.
        alpha : Float, optional
            Adjusts the hyperbolic curvature. Values larger than one yield a
            more distorted embedding where points are pushed to the outer
            boundary (i.e. the ideal points) of hyperblic space. The
            interaction between code{curvature} and code{alpha} is non-linear.
            The default is 1.1.
        equi_adj : Float, optional
            Equi-angular adjustment; must be a real number between zero and
            one; only used if dim is 2. Value 0 means no ajustment, 1
            adjusts embedded data points such that their angular coordinates
            in the Poincare disc are uniformly distributed. Other values
            interpolate between the two extremes. Setting the parameter to non-
            zero values can make the embedding result look more harmoniuous in
            plots. The default is 0.5.
        **kwargs :
            polar :
                Return polar coordinates in dimension 2. This flag is
                ignored in higher dimension).
            isotropic_adj :
                Perform isotropic adjustment, ignoring Eigenvalues
                (default: TRUE if dim is 2, FALSE else)
            lorentz :
                Return raw Lorentz coordinates (before projection to
                hyperbolic space) (default: FALSE)
            stress :
                Return embedding stress


        Yields
        ------
        An dictionary with:
            r : ndarray
                1D array of the radii of the embeded points
            direction : ndarray
                dim-1 array of the directions of the embedded points
            theta : ndarray
                1D array of the polar coordinate angles of the embedded points
                only if embedded into 2D Poincare disk
            stress : float
                The stress of the embedding

        """

        # sanitize/check input
        if(any(np.diag(D) != 0)):  # non-zero diagonal elements are set to zero
            np.fill_diagonal(D, 0)
            warnings.warn("Diagonal of input matrix D has been set to zero")

        if not np.allclose(D, np.transpose(D)):
            warnings.warn(
                "Input matrix D is not symmetric.\
                    Lower triangle part is used.")

        if dim == 2:
            # set default values in dimension 2
            if "isotropic_adj" in kwargs:
                kwargs.isotropic_adj = True
            if "polar" in kwargs:
                kwargs.polar = True
        else:
            # set default values in dimension > 2
            if "isotropic_adj" in kwargs:
                kwargs.isotropic_adj = False
            if "polar" in kwargs:
                warnings.warn("Polar coordinates only valid in dimension two")
                kwargs.polar = False
            if equi_adj != 0.0:
                warnings.warn(
                    "Equiangular adjustment only possible in dimension two.")

        # convert distance matrix to 'hyperbolic Gram matrix'
        A = np.cosh(np.sqrt(curvature)*D)
        n = A.shape[0]

        # check for large/infinite values
        A_max = np.amax(A)
        if(A_max > 1e8):
            warnings.warn(
                "Gram Matrix contains values > 1e8. Rerun with smaller\
                curvature parameter or rescaled distances.")
        if A_max == float("inf"):
            warnings.warn(
                "Gram matrix contains infinite values.\
                Rerun with smaller curvature parameter or rescaled distances.")

        # Compute Eigendecomposition of A
        w, v = np.linalg.eigh(A)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        # Extract leading Eigenvalue and Eigenvector
        lambda0 = w[0]
        x0 = v[:, 0]

        # Extract lower tail of spectrum)
        X = v[:, (n-dim):n]  # Last dim Eigenvectors
        spec_tail = w[(n-dim):n]  # Last dim Eigenvalues
        # A_frob = np.sqrt(np.sum(v**2)) # Frobenius norm of A

        x0 = x0 * np.sqrt(lambda0)  # scale by Eigenvalue
        if(x0[0] < 0):
            x0 = -x0  # Flip sign if first element negative
        x_min = min(x0)  # find minimum

        # no isotropic adjustment: rescale Eigenvectors by Eigenvalues
        if not kwargs.get('isotropic_adj'):
            if np.array([spec_tail > 0]).any():
                warnings.warn(
                    "Spectral Values have been truncated to zero. Try to use\
                    lower embedding dimension")
                spec_tail[spec_tail > 0] = 0
            X = np.matmul(X, np.diag(np.sqrt(-spec_tail)))

        s = np.sqrt(np.sum(X**2, axis=1))
        directional = X / s[:, None]  # convert to directional coordinates

        output = {}  # Allocate output list

        # Calculate radial coordinate
        # multiplicative adjustment (scaling)
        r = np.sqrt((alpha*x0 - x_min)/(alpha*x0 + x_min))
        output['r'] = r

        # Calculate polar coordinates if dimension is 2
        if(dim == 2):
            # calculate polar angle
            theta = np.arctan2(X[:, 0], -X[:, 1])

            # Equiangular adjustment
            if equi_adj > 0.0:
                angles = [(2*x/n-1)*math.pi for x in range(0, n)]
                theta_equi = np.array([x for _, x in sorted(
                    zip(theta, angles))])  # Equi-spaced angles
                # convex combination of original and equi-spaced angles
                theta = (1-equi_adj)*theta + equi_adj*theta_equi
                # update directional coordinate
                directional = np.array(
                    [np.cos(theta), np.sin(theta)]).transpose()

                output['theta'] = theta

        output['directional'] = directional

        # Set Additional return values
        if kwargs.get('lorentz'):
            output['x0'] = x0
            output['X'] = X

        if kwargs.get('stress'):
            output['stress'] = utilFunc.stress(r, directional, curvature, D)

        output['curvature'] = curvature
        output['dim'] = dim
        return(output)

    @staticmethod
    def stress(r, directional, curvature, D):
        # Calculate stress of embedding from radial/directional coordinate
        # From Nagano 2019
        n = len(r)  # number of embeded points
        stress_sq = 0.0  # allocate squared stress

        # convert from numpy to torch
        dist = torch.zeros((n, n))
        r = torch.from_numpy(r)
        directional = torch.from_numpy(directional)
        curvature = torch.tensor(curvature, dtype=torch.double)
        D = torch.tensor(D)

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = utilFunc.hyperbolic_distance(
                        r[i], r[j],
                        directional[i, ], directional[j, ],
                        curvature)
                    stress_sq = stress_sq + (dist[i][j] - D[i, j]) ** 2

        return np.sqrt(stress_sq)
    
    @staticmethod
    def post_order_traversal(mst, currentNode, peel, visited):
        """Post-order traversal of a constrained-MST

        Args:
            mst ([type]): [description]
            currentNode ([type]): [description]
            peel ([type]): [description]
            visited ([type]): [description]

        Returns:
            [type]: [description]
        """
        visited[currentNode] = True
        if mst[currentNode].__len__() < 2:      # leaf nodes
            return currentNode
        else:                                   # internal nodes
            childs = []
            for child in mst[currentNode]:
                if(not visited[child]):
                    childs.append(utilFunc.post_order_traversal(mst, child, peel, visited))
                    # childs.append(child)
            childs.append(currentNode)
            peel.append(childs)
            return currentNode
    
    @staticmethod
    def hyperbolic_distance(r1, directional1, r2, directional2, curvature):
        """Generates hyperbolic distance between two points in poincoire ball

        Args:
            r1 (tensor): radius of point 1
            r2 (tensor): radius of point 2
            directional1 (1D tensor): directional of point 1
            directional2 (1D tensor): directional of point 2
            curvature (integer): curvature

        Returns:
            tensor: distance between point 1 and point 2
        """
        iprod = torch.clamp(torch.dot(directional1, directional2), min=-1.0, max=1.0)
        acosharg = 1.0 + torch.clamp(2.0 * (torch.pow(r1, 2) + torch.pow(r2, 2) - 2 * r1 *
                                            r2 * iprod) / ((1 - torch.pow(r1, 2)) * (1 - torch.pow(r2, 2))), min=0.0)
        # hyperbolic distance between points i and j
        return 1. / torch.sqrt(curvature) * torch.acosh(acosharg)

    @staticmethod
    def make_peel(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
        """Create a tree represtation (peel) from its hyperbolic embedic data

        Args:
            leaf_r (1D tensor): radius of the leaves
            leaf_dir (2D tensor): directional tensors of leaves 
            int_r (1D tensor): radius of internal nodes
            int_dir (2D tensor): directional tensors of internal nodes
        """
        leaf_node_count = leaf_r.shape[0]
        node_count = leaf_r.shape[0] + int_r.shape[0]
        edge_list = defaultdict(list)

        # edge_list = np.array(edge_list, dtype=u_edge)

        for i in range(node_count):
            for j in range(max(i+1, leaf_node_count), node_count):
                dist_ij = 0

                if(i < leaf_node_count):
                    # leaf to internal
                    dist_ij = utilFunc.hyperbolic_distance(
                        leaf_r[i],
                        leaf_dir[i],
                        int_r[j-leaf_node_count],
                        int_dir[j - leaf_node_count],
                        curvature)
                else:
                    # internal to internal
                    i_node = i - leaf_node_count
                    dist_ij = utilFunc.hyperbolic_distance(
                        int_r[i_node],
                        int_dir[i_node],
                        int_r[j-leaf_node_count],
                        int_dir[j - leaf_node_count],
                        curvature)

                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))

                edge_list[i].append(u_edge(dist_ij, i, j))
                edge_list[j].append(u_edge(dist_ij, j, i))

        # construct a minimum spanning tree among the internal nodes
        queue = []  # queue here is a min-heap
        heapify(queue)
        visited = node_count*[False]  # visited here is a boolen list
        heappush(queue, u_edge(0,0,0))    # add a start_edge
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
                    if not found_internal:
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
        visited = (node_count+1) * [False] # all nodes + the fake root 
        utilFunc.post_order_traversal(mst_adjacencies, fake_root, peel, visited)

        for i in range(peel.__len__()):
            for j in range(peel[i].__len__()):
                peel[i][j] += 1     # node re-indexing (1-based)

        return np.array(peel)
