import numpy as np
import torch

from dodonaphy import poincare, Chyp_np
from dodonaphy.edge import Edge
from dendropy.calculate.phylogeneticdistance import NodeDistanceMatrix


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
    # Ensure directional is unit vector
    if not torch.allclose(
        torch.norm(directional, dim=-1).double(), torch.tensor(1.0).double()
    ):
        raise RuntimeError("Directional given is not a unit vector.")

    if r.shape == torch.Size([]):
        return directional * r
    return directional * r[:, None]


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
    n_leaf = leaf_dir.shape[0]
    n_points = n_leaf + int_r.shape[0]
    X = torch.zeros((n_points + 1, dim))  # extra point for root
    if leaf_r.shape == torch.Size([]):
        leaf_r = torch.tile(leaf_r, (n_leaf,))

    X[:n_leaf, :] = dir_to_cart(leaf_r, leaf_dir)
    X[n_leaf:-1, :] = dir_to_cart(int_r, int_dir)

    # fake root node is above node 0
    X[-1, :] = dir_to_cart(leaf_r[0], leaf_dir[0])

    return X


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
    r = torch.pow(torch.pow(X, 2).sum(dim=1), 0.5)
    directional = X / r[:, None]

    for i in torch.where(torch.isclose(r, torch.zeros_like(r))):
        directional[i, 0] = 1
        directional[i, 1:] = 0

    if np_flag:
        r.detach().numpy()
        directional.detach().numpy()

    return r, directional


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

    (leaf_r, leaf_dir) = cart_to_dir(X[:S, :])
    (int_r, int_dir) = cart_to_dir(X[S:, :])

    return leaf_r, int_r, leaf_dir, int_dir


def get_plca(locs, as_torch=False):
    """Return a pair-wise least common ancestor matrix based.

    Args:
        locs ([type]): Coordinates in the Poincare ball
        as_torch (bool): to return as torch tensor (otherwise list).

    Returns:
        [type]: A list of lists containing the edges.
        The "distance" of each edge is the negative of the distance
        of the LCA to the origin.
    """
    node_count = locs.shape[0]
    if as_torch:
        edge_adj = torch.zeros((node_count, node_count), dtype=torch.double)
    else:
        edge_list = [[] for _ in range(node_count)]

    for i in range(node_count):
        for j in range(i):
            dist_ij = poincare.hyp_lca(locs[i], locs[j], return_coord=False)
            if as_torch:
                edge_adj[i, j] = dist_ij
                edge_adj[j, i] = dist_ij
            else:
                edge_list[i].append(Edge(dist_ij, i, j))
                edge_list[j].append(Edge(dist_ij, j, i))

    if as_torch:
        return edge_adj
    return edge_list


def get_plca_np(locs, as_numpy=False):
    """Return a pair-wise least common ancestor matrix based.

    Args:
        locs ([type]): Coordinates on the Hyperboloid.

    Returns:
        [type]: A list of lists containing the edges.
        The "distance" of each edge is the negative of the distance
        of the LCA to the origin.
    """
    node_count = locs.shape[0]
    if as_numpy:
        edge_adj = np.zeros((node_count, node_count), dtype=np.double)
    else:
        edge_list = [[] for _ in range(node_count)]

    for i in range(node_count):
        for j in range(i):
            poin_i = Chyp_np.hyper_to_poincare(locs[i])
            poin_j = Chyp_np.hyper_to_poincare(locs[j])
            dist_ij = poincare.hyp_lca_np(poin_i, poin_j, return_coord=False)
            if as_numpy:
                edge_adj[i, j] = dist_ij
                edge_adj[j, i] = dist_ij
            else:
                edge_list[i].append(Edge(dist_ij, i, j))
                edge_list[j].append(Edge(dist_ij, j, i))

    if as_numpy:
        return edge_adj
    return edge_list


def normalise(y):
    """Normalise vectors to unit sphere

    Args:
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    if y.ndim == 1:
        y = y.unsqueeze(dim=-1)
    dim = y.shape[1]

    norm = torch.norm(y, dim=-1, keepdim=True).repeat(1, dim)
    return y / norm


def normalise_LADJ(y):
    """Copmute log of absolute value of determinate of jacobian of normalising to a directional

    Args:
        y (tensor): Points in R^n n_points x n_dims

    Returns:
        tensor: Jacobain matrix
    """
    if y.ndim == 1:
        y = y.unsqueeze(dim=-1)
    n, D = y.shape

    # norm for each point
    norm = torch.norm(y, dim=-1, keepdim=True)

    log_abs_det_J = torch.zeros(1)
    for k in range(n):
        J = torch.div(
            torch.eye(D, D) - torch.div(torch.outer(y[k], y[k]), torch.pow(norm[k], 2)),
            norm[k],
        )
        log_abs_det_J = log_abs_det_J + torch.logdet(J)
    return log_abs_det_J


def LogDirPrior(blen, aT, bT, a, c):

    n_branch = int(len(blen))
    n_leaf = int(n_branch / 2 + 1)

    treeL = torch.sum(blen)

    if not torch.all(blen > 0.0):
        Warning("Branch lengths clampped to be positive.")
        blen = torch.clamp(blen, min=1e-4)

    tipb = torch.sum(torch.log(blen[:n_leaf]))
    intb = torch.sum(torch.log(blen[n_leaf:]))

    lnPrior = (a - 1) * tipb + (a * c - 1) * intb
    lnPrior = lnPrior + (aT - a * n_leaf - a * c * (n_leaf - 3)) * torch.log(treeL) - bT * treeL
    return lnPrior


def tip_distances(tree0):
    """Get tip pair-wise tip distances"""
    n_taxa = len(tree0)
    dists = np.zeros((n_taxa, n_taxa))
    pdc = tree0.phylogenetic_distance_matrix()
    for i, t1 in enumerate(tree0.taxon_namespace[:]):
        for j, t2 in enumerate(tree0.taxon_namespace[: i + 1]):
            dists[i][j] = pdc(t1, t2)
            dists[j][i] = dists[i][j]
    return dists


def all_distances(tree0):
    """Get tip pair-wise node distances."""
    # Create a NodeDistanceMatrix object from the tree
    ndm = NodeDistanceMatrix.from_tree(tree0)

    nodes = ndm._node_phylogenetic_distances.keys()
    num_nodes = len(nodes)
    dists = np.zeros((num_nodes, num_nodes))

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            distance = ndm._node_phylogenetic_distances[node1][node2]
            dists[i, j] = distance
            dists[j, i] = dists[i][j]

    return dists
