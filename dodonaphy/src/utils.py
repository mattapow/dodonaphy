import numpy as np
import torch
from collections import defaultdict
from .hyperboloid import poincare_to_hyper, lorentz_product
from .edge import u_edge
from . import poincare


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


def get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0] + int_r.shape[0]
    edge_list = defaultdict(list)

    for i in range(node_count):
        for j in range(max(i + 1, leaf_node_count), node_count):
            dist_ij = 0

            if (i < leaf_node_count):
                # leaf to internal
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    int_r[j - leaf_node_count],
                    leaf_dir[i],
                    int_dir[j - leaf_node_count],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                dist_ij = hyperbolic_distance(
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


def get_pdm_tips(leaf_r, leaf_dir, curvature=torch.ones(1)):
    leaf_node_count = leaf_r.shape[0]
    edge_list = [[] for _ in range(leaf_node_count)]

    for i in range(leaf_node_count):
        for j in range(i):
            dist_ij = 0
            dist_ij = hyperbolic_distance(
                leaf_r[i], leaf_r[j], leaf_dir[i], leaf_dir[j], curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = torch.log(torch.cosh(dist_ij))

            edge_list[i].append(u_edge(dist_ij, i, j))
            edge_list[j].append(u_edge(dist_ij, j, i))

    return edge_list


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
    r = torch.pow(torch.pow(X, 2).sum(dim=1), .5)
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
    z1 = poincare_to_hyper(dir_to_cart(r1, directional1)).squeeze()
    z2 = poincare_to_hyper(dir_to_cart(r2, directional2)).squeeze()
    eps = torch.finfo(torch.float64).eps
    inner = torch.clamp(-lorentz_product(z1, z2), min=1+eps)
    return 1. / torch.sqrt(curvature) * torch.acosh(inner)


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


def get_plca(locs):
    """Return a pair-wise least common ancestor matrix based.

    Args:
        locs ([type]): Coordinates in the Poincare ball

    Returns:
        [type]: A list of lists containing the edges.
    """
    node_count = locs.shape[0]
    edge_list = [[] for _ in range(node_count)]

    for i in range(node_count):
        for j in range(i):
            dist_ij = poincare.hyp_lca(locs[i], locs[j], return_coord=False)

            edge_list[i].append(u_edge(dist_ij, i, j))
            edge_list[j].append(u_edge(dist_ij, j, i))

    return edge_list


def ball2real(loc_ball, radius=1):
    """A map from the unit ball B^n to real R^n.
    Inverse of real2ball.

    Args:
        loc_ball (tensor): [description]
        radius (tensor): [description]

    Returns:
        tensor: [description]
    """
    dim = loc_ball.shape[1]
    norm_loc_ball = torch.norm(loc_ball, dim=-1, keepdim=True).repeat(1, dim)
    loc_real = loc_ball / (radius - norm_loc_ball)
    return loc_real


def real2ball(loc_real, radius=1):
    """A map from the reals R^n to unit ball B^n.
    Inverse of ball2real.

    Args:
        loc_real (tensor): Point in R^n with size [1, dim]
        radius (float): Radius of ball

    Returns:
        tensor: [description]
    """
    if loc_real.ndim == 1:
        loc_real = loc_real.unsqueeze(dim=-1)
    dim = loc_real.shape[1]

    norm_loc_real = torch.norm(loc_real, dim=-1, keepdim=True).repeat(1, dim)
    loc_ball = torch.div(radius * loc_real, (1 + norm_loc_real))
    return loc_ball


def real2ball_LADJ(y, radius=1):
    """Copmute log of absolute value of determinate of jacobian of real2ball on point y

    Args:
        y (tensor): Points in R^n n_points x n_dimensions

    Returns:
        scalar tensor: log absolute determinate of Jacobian
    """
    # TODO: move to cython
    if y.ndim == 1:
        y = y.unsqueeze(dim=-1)

    n, D = y.shape
    log_abs_det_J = torch.zeros(1)

    norm = torch.norm(y, dim=-1, keepdim=True)

    for k in range(n):
        J = (torch.eye(D, D) - torch.outer(y[k], y[k]) / (norm[k] * (norm[k] + 1))) / (1+norm[k])
        log_abs_det_J = log_abs_det_J + torch.logdet(radius * J)

    return log_abs_det_J


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
        J = torch.div(torch.eye(D, D) - torch.div(torch.outer(y[k], y[k]), torch.pow(norm[k], 2)), norm[k])
        log_abs_det_J = log_abs_det_J + torch.logdet(J)
    return log_abs_det_J


def LogDirPrior(blen, aT, bT, a, c):

    n_branch = int(len(blen))
    n_leaf = int(n_branch / 2 + 1)

    treeL = sum(blen)

    blen_pos = blen.clone()
    blen_pos[np.isclose(blen_pos, 0)] = 1

    tipb = torch.sum(torch.log(blen_pos[:n_leaf]))
    intb = torch.sum(torch.log(blen_pos[n_leaf:]))

    lnPrior = (a-1)*tipb + (a*c-1)*intb
    lnPrior = lnPrior + (aT - a*n_leaf - a*c*(n_leaf-1)) * torch.log(treeL) - bT*treeL

    return lnPrior
