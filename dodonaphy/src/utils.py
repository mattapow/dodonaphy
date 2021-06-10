import math
from heapq import heapify, heappush, heappop
from collections import deque
import numpy as np
import torch
import warnings
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from .hyperboloid import poincare_to_hyper, lorentz_product


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
        if any(np.diag(D) != 0):  # non-zero diagonal elements are set to zero
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
        A = np.cosh(np.sqrt(curvature) * D)
        n = A.shape[0]

        # check for large/infinite values
        A_max = np.amax(A)
        if A_max > 1e8:
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
        X = v[:, (n - dim):n]  # Last dim Eigenvectors
        spec_tail = w[(n - dim):n]  # Last dim Eigenvalues
        # A_frob = np.sqrt(np.sum(v**2)) # Frobenius norm of A

        x0 = x0 * np.sqrt(lambda0)  # scale by Eigenvalue
        if x0[0] < 0:
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

        s = np.sqrt(np.sum(X ** 2, axis=1))
        directional = X / s[:, None]  # convert to directional coordinates

        output = {}  # Allocate output list

        # Calculate radial coordinate
        # multiplicative adjustment (scaling)
        r = np.sqrt((alpha * x0 - x_min) / (alpha * x0 + x_min))
        output['r'] = r

        # Calculate polar coordinates if dimension is 2
        if dim == 2:
            # calculate polar angle
            theta = np.arctan2(X[:, 0], -X[:, 1])

            # Equiangular adjustment
            if equi_adj > 0.0:
                angles = [(2 * x / n - 1) * math.pi for x in range(0, n)]
                theta_equi = np.array([x for _, x in sorted(
                    zip(theta, angles))])  # Equi-spaced angles
                # convex combination of original and equi-spaced angles
                theta = (1 - equi_adj) * theta + equi_adj * theta_equi
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
        return output

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
        if mst[currentNode].__len__() < 2:  # leaf nodes
            return currentNode
        else:  # internal nodes
            childs = []
            for child in mst[currentNode]:
                if (not visited[child]):
                    childs.append(utilFunc.post_order_traversal(
                        mst, child, peel, visited))
                    # childs.append(child)
            childs.append(currentNode)
            peel.append(childs)
            return currentNode

    def angle_to_directional(theta):
        """
        Convert polar angles to unit vectors

        Parameters
        ----------
        theta : tensor
            Angle of points.

        Returns
        -------
        directional : tensor
            Unit vectors of points.

        """
        dim = 2
        n_points = len(theta)
        directional = torch.zeros(n_points, dim)
        directional[:, 0] = torch.cos(theta)
        directional[:, 1] = torch.sin(theta)
        return directional

    @staticmethod
    def make_peel(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1), method='mst'):
        if method == 'mst':
            return utilFunc.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature)
        elif method == 'geodesics':
            leaf_locs = utilFunc.dir_to_cart(leaf_r, leaf_dir)
            return utilFunc.make_peel_geodesics(leaf_locs)
        else:
            raise KeyError("Invalid method key. Use 'mst' or 'geodesics'.")

    @staticmethod
    def get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
        leaf_node_count = leaf_r.shape[0]
        node_count = leaf_r.shape[0] + int_r.shape[0]
        edge_list = defaultdict(list)

        for i in range(node_count):
            for j in range(max(i + 1, leaf_node_count), node_count):
                dist_ij = 0

                if (i < leaf_node_count):
                    # leaf to internal
                    dist_ij = utilFunc.hyperbolic_distance(
                        leaf_r[i],
                        int_r[j - leaf_node_count],
                        leaf_dir[i],
                        int_dir[j - leaf_node_count],
                        curvature)
                else:
                    # internal to internal
                    i_node = i - leaf_node_count
                    dist_ij = utilFunc.hyperbolic_distance(
                        int_r[i_node],
                        int_r[j - leaf_node_count],
                        int_dir[i_node],
                        int_dir[j - leaf_node_count],
                        curvature)

                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))

                edge_list[i].append(u_edge(dist_ij, i, j))
                edge_list[j].append(u_edge(dist_ij, j, i))

        return edge_list

    @staticmethod
    def get_pdm_tips(leaf_r, leaf_dir, curvature=torch.ones(1)):
        leaf_node_count = leaf_r.shape[0]
        edge_list = [[] for _ in range(leaf_node_count)]

        for i in range(leaf_node_count):
            for j in range(i):
                dist_ij = 0
                dist_ij = utilFunc.hyperbolic_distance(
                    leaf_r[i], leaf_r[j], leaf_dir[i], leaf_dir[j], curvature)

                # apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))

                edge_list[i].append(u_edge(dist_ij, i, j))
                edge_list[j].append(u_edge(dist_ij, j, i))

        return edge_list

    @staticmethod
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
        edge_list = utilFunc.get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1))

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
        utilFunc.post_order_traversal(
            mst_adjacencies, fake_root, peel, visited)

        return np.array(peel, dtype=np.intc)

    @staticmethod
    def dendrophy_to_pb(tree):
        """ Convert Dendrophy tree to peels and blens.
        Parameters
        ----------
        tree : A Dendrophy tree
        Returns
        -------
        peel : adjacencies of internal nodes (left child, right child, node)
        blens : branch lengths
        """
        # Get branch lengths
        S = len(tree)
        n_edges = 2 * S - 2
        blens = torch.zeros(n_edges)
        for i in range(n_edges):
            blens[i] = tree.bipartition_edge_map[tree.bipartition_encoding[i]].length

        # Get peel
        nds = [nd for nd in tree.postorder_internal_node_iter()]
        n_int_nds = len(nds)
        peel = np.zeros((n_int_nds, 3), dtype=int)
        for i in range(n_int_nds):
            peel[i, 0] = tree.bipartition_encoding.index(nds[i].child_edges()[0].bipartition)
            peel[i, 1] = tree.bipartition_encoding.index(nds[i].child_edges()[1].bipartition)
            peel[i, 2] = tree.bipartition_encoding.index(nds[i].bipartition)
        return peel, blens.double()

    @staticmethod
    def dir_to_cart(r, directional):
        """convert radius/ directionals to cartesian coordinates [x,y,z,...]

        Parameters
        ----------
        r (1D tensor): radius of each n_points
        directional (2D tensor): n_points x dim directional of each point

        Returns
        -------
        (2D tensor) Cartesian coordinates of each point n_points x dim

        """
        # # Ensure directional is unit vector
        # if not torch.allclose(torch.norm(directional, dim=-1).double(), torch.tensor(1.).double()):
        #     raise RuntimeError('Directional given is not a unit vector.')

        if r.shape == torch.Size([]):
            return directional * r
        return directional * r[:, None]

    @staticmethod
    def dir_to_cart_tree(leaf_r, int_r, leaf_dir, int_dir, dim):
        """Convert radius/ directionals to cartesian coordinates [x,y] from tree data

        Parameters
        ----------
        leaf_r
        int_r
        leaf_dir
        int_dir
        dim

        Returns
        -------
        2D tensor: Cartesian coords of leaves, then internal nodes, then root above node 0

        """
        n_leaf = leaf_r.shape[0]
        n_points = n_leaf + int_r.shape[0]
        X = torch.zeros((n_points + 1, dim))  # extra point for root

        X[:n_leaf, :] = utilFunc.dir_to_cart(leaf_r, leaf_dir)
        X[n_leaf:-1, :] = utilFunc.dir_to_cart(int_r, int_dir)

        # fake root node is above node 0
        X[-1, :] = utilFunc.dir_to_cart(leaf_r[0], leaf_dir[0])

        return X

    @staticmethod
    def cart_to_dir(X):
        """convert positions in X in  R^dim to radius/ unit directional

        Parameters
        ----------
        X (2D tensor or ndarray): Points in R^dim

        Returns
        -------
        r (Tensor): radius
        directional (Tensor): unit vectors

        """
        np_flag = False
        if type(X).__module__ == np.__name__:
            np_flag = True
            X = torch.from_numpy(X)

        if X.ndim == 1:
            X = torch.unsqueeze(X, 0)
        r = torch.pow(torch.pow(X, 2).sum(dim=1), .5)
        directional = X / r[:, None]

        for i in torch.where(torch.isclose(r, torch.zeros_like(r))):
            directional[i, 0] = 1
            directional[i, 1:] = 0

        if np_flag:
            r.detach().numpy()
            directional.detach().numpy()

        return r, directional

    @staticmethod
    def cart_to_dir_tree(X):
        """Convert Cartesian coordinates in R^2 to radius/ unit directional

        Parameters
        ----------
        X (2D Tensor or ndarray): Cartesian coordinates of leaves and then internal nodes n_points x dim

        Returns (Tensor)
        -------
        Location of leaves and internal nodes separately using r, dir
        """

        S = int(X.shape[0] / 2 + 1)

        (leaf_r, leaf_dir) = utilFunc.cart_to_dir(X[:S, :])
        (int_r, int_dir) = utilFunc.cart_to_dir(X[S:, :])

        return leaf_r, int_r, leaf_dir, int_dir

    @staticmethod
    def plot_tree(ax, peel, X, color=(0, 0, 0), labels=True):
        """ Plot a tree in the Poincare disk

        Parameters
        ----------
        ax (axes): Axes for plotting
        peel (2D Tensor): edges
        X (2D Tensor): node positions [x, y]
        color (tuple): rgb colour

        Returns
        -------

        """
        circ = Circle((0, 0), radius=1, fill=False, edgecolor='k')
        ax.add_patch(circ)

        # nodes
        ax.plot(X[:, 0], X[:, 1], '.', color=color)

        # edges
        n_parents = peel.shape[0]
        for i in range(n_parents):
            left = peel[i, 0]
            right = peel[i, 1]
            parent = peel[i, 2]
            # TODO: correctly curved lines, not straight lines
            line = Line2D([X[left, 0], X[parent, 0]],
                          [X[left, 1], X[parent, 1]],
                          linewidth=1,
                          color=color)
            ax.add_line(line)
            line = Line2D([X[right, 0], X[parent, 0]],
                          [X[right, 1], X[parent, 1]],
                          linewidth=1,
                          color=color)
            ax.add_line(line)

        if labels:
            n_points = X.shape[0] - 1
            for p in range(n_points):
                msg = str(p)
                if p == 0:
                    msg = msg + " (" + str(n_points) + ")"
                ax.annotate(msg,
                            xy=(float(X[p, 0]) + .04, float(X[p, 1])),
                            xycoords='data')

    @staticmethod
    def hyperbolic_distance(r1, r2, directional1, directional2, curvature):
        """Generates hyperbolic distance between two points in poincoire ball

        Args:
            r1 (tensor): radius of point 1
            r2 (tensor): radius of point 2
            directional1 (1D tensor): directional of point 1
            directional2 (1D tensor): directional of point 2
            curvature (tensor): curvature

        Returns:
            tensor: distance between point 1 and point 2
        """
        # if torch.allclose(r1, r2) and torch.allclose(directional1, directional2):
        #     return torch.zeros(1)

        # Use lorentz distance for numerical stability
        z1 = poincare_to_hyper(utilFunc.dir_to_cart(r1, directional1)).squeeze()
        z2 = poincare_to_hyper(utilFunc.dir_to_cart(r2, directional2)).squeeze()
        eps = torch.finfo(torch.float64).eps
        inner = torch.clamp(-lorentz_product(z1, z2), min=1+eps)
        return 1. / torch.sqrt(curvature) * torch.acosh(inner)

    @staticmethod
    def hyperbolic_distance_locs(z1, z2, curvature=torch.ones(1)):
        """Generates hyperbolic distance between two points in poincoire ball

        Args:
            z1 (tensor): coords or point 1 in Poincare ball
            z2 (tensor): coords or point 2 in Poincare ball
            curvature (tensor): curvature

        Returns:
            tensor: distance between point 1 and point 2
        """

        # Use lorentz distance for numerical stability
        z1 = poincare_to_hyper(z1).squeeze()
        z2 = poincare_to_hyper(z2).squeeze()
        eps = torch.finfo(torch.float64).eps
        inner = torch.clamp(-lorentz_product(z1, z2), min=1+eps)
        return 1. / torch.sqrt(curvature) * torch.acosh(inner)

    @staticmethod
    def tree_to_newick(tipnames, peel_row, blen_row):
        """This function returns a Tree in newick format from given peel and branch lengths

        Args:
            tipnames (List): List of Taxa labels
            peel_row (List of List): Peel in post-order indexing
            blen_row (List): Branch lengths

        Returns:
            Newick String: A Tree
        """
        chunks = {}
        plen = tipnames.__len__()-1
        for p in range(plen):
            n1 = peel_row[p][0]
            n2 = peel_row[p][1]
            n3 = peel_row[p][2]
            if n1 <= plen:
                chunks[n1] = tipnames[n1] + ":" + str(blen_row[n1].item())
            if n2 <= plen:
                chunks[n2] = tipnames[n2] + ":" + str(blen_row[n2].item())

            if p == (plen-1):
                chunks[n3] = "(" + chunks[n1] + "," + chunks[n2] + ")" + ";"
            else:
                chunks[n3] = "(" + chunks[n1] + "," + chunks[n2] + ")" + ":" + str(blen_row[n3].item())
        return str(chunks[peel_row[-1][2]])

    @staticmethod
    def save_tree_head(path_write, filename, S):
        fn = path_write + '/' + filename + '.trees'
        with open(fn, 'w') as file:
            file.write("#NEXUS\n\n")
            file.write("Begin taxa;\n\tDimensions ntax=" + str(S) + ";\n")
            file.write("\tTaxlabels\n")
            for i in range(S):
                file.write("\t\t" + "T" + str(i+1) + "\n")
            file.write("\t\t;\nEnd;\n\n")
            file.write("Begin trees;\n")

    @staticmethod
    def save_tree(dir, filename, peel, blens, iteration, LL):
        S = len(peel)+1
        tipnames = ['T' + str(x+1) for x in range(S)]
        tree = utilFunc.tree_to_newick(tipnames, peel, blens)

        fn = dir + '/' + filename + '.trees'
        with open(fn, 'a+') as file:
            file.write("tree STATE_" + str(iteration))
            # TODO: output likelihoods
            file.write(" [&lnP={}] = [&R] ".format(LL))
            file.write(tree + '\n')

    @staticmethod
    def make_peel_geodesics(leaf_locs):
        # /**
        #  * use geodesic arcs to make a binary tree from a set of leaf node points embedded
        #  * in hyperbolic space.
        #  * Output: int_locs, peel
        #  */
        dims = leaf_locs.shape[1]
        leaf_node_count = leaf_locs.shape[0]
        int_node_count = leaf_locs.shape[0] - 2
        node_count = leaf_locs.shape[0] * 2 - 2
        leaf_r, leaf_dir = utilFunc.cart_to_dir(leaf_locs)
        leaf_r = torch.cat((leaf_r, leaf_r[0].unsqueeze(dim=0)))
        leaf_dir = torch.cat((leaf_dir, leaf_dir[-1, :].unsqueeze(dim=0)), dim=0)
        int_locs = torch.zeros(int_node_count+1, dims)
        int_locs[-1, :] = leaf_locs[0, :]
        edge_list = utilFunc.get_pdm_tips(leaf_r, leaf_dir, curvature=torch.ones(1))
        peel = np.zeros((int_node_count+1, 3), dtype=np.int16)

        # queue = [edges for neighbours in edge_list for edges in neighbours]
        queue = []
        heapify(queue)
        heappush(queue, min(edge_list[0]))
        visited = node_count * [False]
        print('int_node_count: %d\n' % int_node_count)
        int_i = 0
        while int_i < int_node_count:
            print('int_i = %d' % int_i)
            e = queue.pop()
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

            int_locs[int_i] = utilFunc.hyp_lca(from_point, to_point)
            cur_peel = cur_internal - leaf_node_count
            peel[cur_peel][0] = e.from_
            peel[cur_peel][1] = e.to_
            peel[cur_peel][2] = cur_internal
            visited[e.from_] = True
            visited[e.to_] = True

            print(peel)
            # print(e)
            # print(visited)
            # print(int_locs)
            # print(int_i)
            print('done')

            # add all pairwise distances between the new node and other active nodes
            for i in range(cur_internal):
                if visited[i]:
                    continue
                if i < leaf_node_count:
                    dist_ij = utilFunc.hyperbolic_distance_locs(leaf_locs[i], int_locs[int_i])
                else:
                    dist_ij = utilFunc.hyperbolic_distance_locs(int_locs[i-leaf_node_count], int_locs[int_i])
                #  // apply the inverse transform from Matsumoto et al 2020
                dist_ij = torch.log(torch.cosh(dist_ij))
                # // use negative of distance so that least dist has largest value in the priority queue
                heappush(queue, u_edge(-dist_ij, i, cur_internal))
            int_i += 1

        # add fake root
        int_locs[-1, :] = leaf_locs[0, :]
        child = peel[0][2]
        peel[-1, :] = [0, child, node_count]

        return peel, int_locs

    @staticmethod
    def isometric_transform(a, x):
        """Reflection (circle inversion of x through orthogonal circle centered at a)."""
        r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
        u = x - a
        return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a

    @staticmethod
    def reflection_center(mu):
        """Center of inversion circle."""
        return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)

    @staticmethod
    def euc_reflection(x, a):
        """
        Euclidean reflection (also hyperbolic) of x
        Along the geodesic that goes through a and the origin
        (straight line)
        """
        MIN_NORM = 1e-15
        xTa = torch.sum(x * a, dim=-1, keepdim=True)
        norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        proj = xTa * a / norm_a_sq
        return 2 * proj - x

    @staticmethod
    def _halve(x):
        """ computes the point on the geodesic segment from o to x at half the distance """
        return x / (1. + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))

    @staticmethod
    def hyp_lca(a, b, return_coord=True):
        """
        Computes projection of the origin on the geodesic between a and b, at scale c
        More optimized than hyp_lca1
        """
        r = utilFunc.reflection_center(a)
        b_inv = utilFunc.isometric_transform(r, b)
        o_inv = a
        o_inv_ref = utilFunc.euc_reflection(o_inv, b_inv)
        o_ref = utilFunc.isometric_transform(r, o_inv_ref)
        proj = utilFunc._halve(o_ref)
        if not return_coord:
            return utilFunc.hyp_dist_o(proj)
        else:
            return proj

    @staticmethod
    def hyp_dist_o(x):
        """
        Computes hyperbolic distance between x and the origin.
        """
        x_norm = x.norm(dim=-1, p=2, keepdim=True)
        return 2 * torch.arctanh(x_norm)