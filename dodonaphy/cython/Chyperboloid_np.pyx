""" Lorentz model of hyperbolic space

Using a hyperboloid surface H^n sitting in R^n+1. Points x in R^n+1 with
lorentz product <x,x>=-1 are on the hyperboloid surface.

If n=2, then we can stereo-graphically project the surface into the Poincare
disk as they are isomorphic.

This file takes numpy arrays as inputs.

General methodology coming from Nagano 2019
A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning
https://github.com/pfnet-research/hyperbolic_wrapped_distribution
"""

import numpy as np
cimport numpy as np
from dodonaphy import utils, Cutils
from dodonaphy.edge import Edge

cdef np.double_t eps = np.finfo(np.double).eps

cdef class Cu_edge:
    
    cdef readonly double distance
    cdef readonly int from_
    cdef readonly int to_

    def __init__(self, double distance, int node_1, int node_2):
        self.distance = distance
        self.from_ = node_1
        self.to_ = node_2

    def __lt__(self, other):
        return self.distance < other.distance

cpdef parallel_transport(
    np.ndarray[np.double_t, ndim=1] xi,
    np.ndarray[np.double_t, ndim=1] x,
    np.ndarray[np.double_t, ndim=1] y):
    """Transport a vector from the tangent space at the x
    to the tangent space at y

        From Nagano 2019


    Parameters
    ----------
    xi : Tensor
        1D vector to be transported.
    x : Tensor
        1D vector of tangent point in initial tangent space T_x H^n.
    y : Tensor
        1D tensor of tangent point in target tangent space T_y H^n..

    Returns
    -------
        1D tensor in tangent space at T_y.

    """
    cdef np.double_t alpha = -lorentz_product(x, y)
    cdef np.double_t coef = lorentz_product(y, xi) / (alpha + 1)
    return xi + coef * (x + y)


cpdef exponential_map(
    np.ndarray[np.double_t, ndim=1] x,
    np.ndarray[np.double_t, ndim=1] v):
    """Exponential map

    Map a vector v on the tangent space T_x H^n onto the hyperboloid

    Parameters
    ----------
    x : Tensor
        Vector defining tangent space T_x H^n
    v : Tensor
        Vector in tanget space T_x H^n

    Returns
    -------
    Tensor
        Projection of v onto the hyperboloid.

    """
    """Exponential map

    Map a vector v on the tangent space T_x H^n onto the hyperboloid

    Parameters
    ----------
    x : Tensor
        Vector defining tangent space T_x H^n
    v : Tensor
        Vector in tanget space T_x H^n

    Returns
    -------
    Tensor
        Projection of v onto the hyperboloid.

    """
    cdef np.double_t vnorm = np.sqrt(np.maximum(lorentz_product(v, v), eps))
    return np.cosh(vnorm) * x + np.sinh(vnorm) * v / vnorm


cpdef exp_map_inverse(np.ndarray[np.double_t, ndim=1] z, np.ndarray[np.double_t, ndim=1] mu):
    """Inverse of exponential map

    Args:
        z ([type]): Hyperboloid location in R^n+1
        mu ([type]): Tangent point

    Returns:
        [type]: [description]
    """

    cdef np.double_t alpha = np.maximum(-lorentz_product(mu, z), 1+eps)
    cdef np.double_t denom = np.sqrt(np.maximum(alpha ** 2 - 1, eps))
    cdef np.double_t factor = np.arccosh(alpha) / (denom)
    return factor * (z - alpha * mu)


cpdef p2t0(np.ndarray[np.double_t, ndim=2] loc, mu=None, get_jacobian=False):
    """Convert location on Poincare ball to Euclidean space.

    Use stereographic projection of point onto hyperboloid surface in R^n+1,
    then project onto z=1 plane, the tangent plane at the origin.

    Parameters
    ----------
    loc: Location in Poincare ball, loc in R^n with |loc|<1
    mu: Base position of vector on tangent plane at origin.

    Returns
    -------
    Projection of location into the tangent space T_0H^n, which is R^n

    """
    cdef np.ndarray[np.double_t, ndim=2] vec_hyp = poincare_to_hyper_2d(loc)
    if mu is None:
        mu = np.zeros_like(loc)
        mu = up_to_hyper(mu)
    n_loc, dim = np.shape(loc)
    cdef np.ndarray[np.double_t, ndim=1] zero = np.zeros((dim + 1), dtype=np.double)
    zero[0] = 1

    cdef np.ndarray[np.double_t, ndim=2] out = np.zeros_like(loc)
    for i in range(n_loc):
        vec_t0_mu = exp_map_inverse(vec_hyp[i, :], mu[i, :])
        vec_t0_0 = parallel_transport(vec_t0_mu, mu[i, :], zero)
        out[i, :] = vec_t0_0[1:]

    if get_jacobian:
        _, jacobian = t02p(out, mu[:, 1:], get_jacobian=True)
        return out, -jacobian
    return out


cpdef t02p(np.ndarray[np.double_t, ndim=2] x, mu=None, get_jacobian=False):
    """Transform a vector x in Euclidean space to the Poincare disk.

    Take a vector in the tangent space of a hyperboloid at the origin, project it
    onto a hyperboloid surface then onto the poincare ball. Mu is the mean of the
    distribution in the Euclidean space

    Parameters
    ----------
    x (Tensor or ndarray): Position of sample in tangent space at origin. n_points x n_dimensions
            x_1, y_1;
            x_2, y_2;
            ...
    mu (Tensor or ndarray): Mean of distribution in tangent space at origin. Must
        be same size as x. Default at origin.

    Returns
    -------
    Transformed vector x, from tangent plane to Poincare ball

    """
    n_loc = x.shape[0]
    dim = x.shape[1]

    if mu is None:
        mu = np.zeros_like(x)

    cdef np.ndarray[np.double_t, ndim=2] x_poin = np.zeros_like(x)
    cdef np.ndarray[np.double_t, ndim=2] mu_hyp = up_to_hyper(mu)
    cdef np.double_t jacobian = np.zeros(1)
    for i in range(n_loc):
        x_hyp = tangent_to_hyper(mu_hyp[i, :], x[i, :], dim)
        x_poin[i, :] = hyper_to_poincare(x_hyp)
        if get_jacobian:
            jacobian = jacobian + tangent_to_hyper_jacobian(mu_hyp[i, :], x[i, :], dim)
            jacobian = jacobian + hyper_to_poincare_jacobian(x_hyp)

    if get_jacobian:
        return x_poin, jacobian
    else:
        return x_poin


cpdef tangent_to_hyper(np.ndarray[np.double_t, ndim=1] mu, np.ndarray[np.double_t, ndim=1] v_tilde, np.int_t dim):
    """Project a vector onto the hyperboloid

    Project a vector from the origin, v_tilde, in the tangent space at the origin T_0 H^n
    onto the hyperboloid H^n at point mu

    Parameters
    ----------
    mu: Point of new tangent space in R^dim+1
    v_tilde: Vector in R^dim to project
    dim: dimension of hyperbolic space H^n

    Returns
    -------
    A point in R^dim+1 on the hyperboloid

    """
    cdef np.ndarray[np.double_t, ndim=1] mu0 = np.concatenate(([1.0], np.zeros(dim, dtype=np.double)))
    cdef np.ndarray[np.double_t, ndim=1] v = np.concatenate(([1.0], np.squeeze(v_tilde))).astype(np.double)
    cdef np.ndarray[np.double_t, ndim=1] u = parallel_transport(v, mu0, mu)
    cdef np.ndarray[np.double_t, ndim=1] z = exponential_map(mu, u)
    return z


cpdef tangent_to_hyper_jacobian(np.ndarray[np.double_t, ndim=1] mu, np.ndarray[np.double_t, ndim=1] v_tilde, np.int_t dim):
    """Return log absolute of determinate of jacobian for tangent_to_hyper

    Args:
        mu ([type]): [description]
        v_tilde ([type]): [description]
        dim ([type]): [description]

    Returns:
        Scalar tensor: [description]
    """
    cdef np.double_t r = np.linalg.norm(v_tilde)
    return np.log(np.power(np.divide(np.sinh(r), r), dim - 1))


cpdef hyper_to_poincare(np.ndarray[np.double_t, ndim=1] location):
    """
    Take stereographic projection from H^n Hyperboloid in R^n+1 onto the
    Poincare ball H^n in R^n

    Parameters
    ----------
    location : Tensor
        1D tensor giving position of a point on Hyperboloid in R^n+1.

    Returns
    -------
    Tensor
        A 1D tensor corresponding to a point in the Poincare ball in R^n.

    """
    cdef np.int_t dim = location.shape[0] - 1
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(dim)
    for i in range(dim):
        out[i] = location[i + 1] / (1 + location[0])
    return out


cpdef hyper_to_poincare_jacobian(np.ndarray[np.double_t, ndim=1] location):
    cdef np.int_t dim = location.shape[0]

    cdef np.double_t a = 1 + location[0]
    cdef np.double_t norm = np.sum(np.power(location[1:], 2))
    cdef np.double_t det = np.divide(np.power(a, 2) + norm, np.power(a, 2 * (dim + 1)))

    # compute Jacobian matrix then get determinant
    # J = np.zeros((dim-1, dim))
    # for i in range(dim-1):
    #     J[i, 0] = - location[i+1] / (1+location[0])**2
    #     J[i, i+1] = 1 / (1 + location[0])
    # det = np.linalg.det(np.power(np.matmul(J.T, J), .5))
    return np.log(np.abs(det))


cpdef up_to_hyper(loc):
    """Project directly up onto the hyperboloid

    Take a position in R^n and return the point in R^n+1 lying on the hyperboloid H^n
    which is directly 'above' it (in the first dimension)

    Parameters
    ----------
    loc: Location in R^n

    Returns
    -------
    Location in R^n+1 on Hyperboloid

    """
    if loc.ndim == 1:
        z = np.expand_dims(np.sqrt(np.sum(np.power(loc, 2)) + 1), 0)
        return np.concatenate((z, loc), axis=0)
    elif loc.ndim == 2:
        z = np.expand_dims(np.sqrt(np.sum(np.power(loc, 2), 1) + 1), 1)
        return np.concatenate((z, loc), axis=1)


cpdef hyperbolic_distance(
    double r1,
    double r2,
    np.ndarray[np.double_t, ndim=1] directional1,
    np.ndarray[np.double_t, ndim=1] directional2,
    np.double_t curvature):
    """Generates hyperbolic distance between two points in poincoire ball

    Args:
        r1 (ndarray): radius of point 1
        r2 (ndarray): radius of point 2
        directional1 (1D ndarray): directional of point 1
        directional2 (1D ndarray): directional of point 2
        curvature (ndarray): curvature

    Returns:
        ndarray: distance between point 1 and point 2
    """
    assert curvature < 0

    # Use lorentz distance for numerical stability
    cdef np.ndarray[np.double_t, ndim=1] z1 = poincare_to_hyper(Cutils.dir_to_cart_np(r1, directional1))
    cdef np.ndarray[np.double_t, ndim=1] z2 = poincare_to_hyper(Cutils.dir_to_cart_np(r2, directional2))
    cdef double inner = np.maximum(-lorentz_product(z1, z2), 1+eps)
    return 1. / np.sqrt(-curvature) * np.arccosh(inner)


cpdef hyperbolic_distance_lorentz(
    np.ndarray[np.double_t, ndim=1] x1,
    np.ndarray[np.double_t, ndim=1]x2,
    np.double_t curvature=-1.0):
    """Generates hyperbolic distance between two points in poincare ball.
    Project onto hyperboloid and compute using Lorentz product.

    Returns:
        tensor: distance between point 1 and point 2
    """

    if np.isclose(curvature, 0.0):
        return np.linalg.norm(x2-x1)

    cdef np.ndarray[np.double_t, ndim=1] z1 = poincare_to_hyper(x1)
    cdef np.ndarray[np.double_t, ndim=1] z2 = poincare_to_hyper(x2)
    cdef np.double_t inner = np.maximum(-lorentz_product(z1, z2), 1.+eps)
    return 1. / np.sqrt(-curvature) * np.arccosh(inner)

cdef lorentz_product(
    np.ndarray[np.double_t, ndim=1] x,
    np.ndarray[np.double_t, ndim=1] y):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : ndarray
        1D array of a point on the hyperboloid.
    y : ndarray optional
        1D array of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    return -x[0] * y[0] + np.dot(x[1:], y[1:])

cdef poincare_to_hyper_2d(np.ndarray[np.double_t, ndim=2] location):
    """
    Take point in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: ndarray
        location of point in poincare ball

    Returns
    -------

    """
    cdef np.int_t n_points = location.shape[0]
    cdef np.int_t dim = location.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] out = np.zeros((n_points, dim + 1))
    cdef np.ndarray[np.double_t, ndim=2] a = np.power(location[:], 2)
    cdef np.ndarray[np.double_t, ndim=2] b = a.sum(axis=1, keepdims=True)
    out[:, 1:] = 2 * location[:] / (1 - b + eps)
    return out

cdef poincare_to_hyper(np.ndarray[np.double_t, ndim=1] location):
    """
    Take point in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: ndarray
        location of point in poincare ball

    Returns
    -------

    """
    cdef np.int_t dim = location.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(dim + 1)
    cdef np.ndarray[np.double_t, ndim=1] a = np.power(location[:], 2)
    cdef np.ndarray[np.double_t, ndim=1] b = a.sum(axis=0, keepdims=True)
    out[0] = (1 + b) / (1 - b + eps)
    out[1:] = 2 * location[:] / (1 - b + eps)
    return out

cpdef ball2real(np.ndarray[np.double_t, ndim=2] loc_ball, np.double_t radius=1.0):
    """A map from the unit ball B^n to real R^n.
    Inverse of real2ball.

    Args:
        loc_ball (tensor): [description]
        radius (tensor): [description]

    Returns:
        tensor: [description]
    """
    if loc_ball.ndim == 1:
        loc_ball = np.expand_dims(loc_ball, -1)
    cdef np.int_t dim = loc_ball.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] norm_loc_ball = np.tile(np.linalg.norm(loc_ball, axis=-1, keepdims=True), (1, dim))
    cdef np.ndarray[np.double_t, ndim=2] loc_real = loc_ball / (radius - norm_loc_ball)
    return loc_real


cpdef real2ball(np.ndarray[np.double_t, ndim=2] loc_real, np.double_t radius=1.0):
    """A map from the reals R^n to unit ball B^n.
    Inverse of ball2real.

    Args:
        loc_real (tensor): Point in R^n with size [1, dim]
        radius (float): Radius of ball

    Returns:
        tensor: [description]
    """
    if loc_real.ndim == 1:
        loc_real = np.expand_dims(loc_real, -1)
    cdef np.int_t dim = loc_real.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] norm_loc_real = np.tile(np.linalg.norm(loc_real, axis=-1, keepdims=True), (1, dim))
    cdef np.ndarray[np.double_t, ndim=2] loc_ball = np.divide(radius * loc_real, (1 + norm_loc_real))
    return loc_ball


cpdef real2ball_LADJ(np.ndarray[np.double_t, ndim=2] y, np.double_t radius=1.0):
    """Copmute log of absolute value of determinate of jacobian of real2ball on point y

    Args:
        y (tensor): Points in R^n n_points x n_dimensions

    Returns:
        scalar tensor: log absolute determinate of Jacobian
    """
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    n, D = np.shape(y)
    cdef np.double_t log_abs_det_J = 0.0
    cdef np.ndarray[np.double_t, ndim=2] norm = np.linalg.norm(y, axis=-1, keepdims=True)
    cdef np.ndarray[np.double_t, ndim=2] J = np.zeros((D, D))
    for k in range(n):
        J = (np.eye(D, D) - np.outer(y[k], y[k]) / (norm[k] * (norm[k] + 1))) / (1+norm[k])
        sign, log_det = np.linalg.slogdet(radius * J)
        log_abs_det_J = log_abs_det_J + sign*log_det
    return log_abs_det_J


cpdef get_pdm_tips(leaf_r, leaf_dir, curvature=-1.0):
    leaf_node_count = leaf_r.shape[0]
    edge_list = [[] for _ in range(leaf_node_count)]

    for i in range(leaf_node_count):
        for j in range(i):
            dist_ij = 0
            dist_ij = hyperbolic_distance(leaf_r[i], leaf_r[j], leaf_dir[i], leaf_dir[j], curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = np.log(np.cosh(dist_ij))

            edge_list[i].append(Edge(dist_ij, i, j))
            edge_list[j].append(Edge(dist_ij, j, i))

    return edge_list

cpdef get_pdm(
    np.ndarray[np.double_t, ndim=1] leaf_r,
    np.ndarray[np.double_t, ndim=2] leaf_dir,
    int_r=None,
    int_dir=None,
    curvature=-np.ones(1),
    dtype='dict'):
    """Pair-wise hyperbolic distance matrix

        Note if curvature=0, then the SQUARED Euclidean distance is computed.
    Args:
        leaf_r (tensor):
        leaf_dir (tensor):
        int_r (1D tensor):
        inr_dir (1D tensor):
        curvature (double): curvature
        dtype (string): "dict" or "numpy"

    Returns:
        ndarray: distance between point 1 and point 2
    """
    cdef np.int_t leaf_node_count = leaf_r.shape[0]
    cdef np.int_t int_node_count = 0
    if int_r is None:
        int_r = np.ndarray((0))
        int_dir = np.ndarray((0, 0))
    else:
        int_node_count = int_r.shape[0]

    cdef np.int_t node_count = leaf_node_count + int_node_count

    # return array if pairwise distance if asNumpy
    cdef asNumpy = dtype == 'numpy'
    cdef np.ndarray[np.double_t, ndim=2] pdm_np = np.zeros((node_count*asNumpy, node_count*asNumpy))

    if np.isclose(curvature, np.zeros(1, dtype=np.double)):
        # Euclidean distance
        X = leaf_r[0] * leaf_dir
        for i in range(X.shape[0]):
            for j in range(X.shape[1]): 
                pdm_np[i, j] = pdm_np[j, i] = np.linalg.norm(X[i, :] - X[j, :])
        return pdm_np

    assert dtype in ('dict', 'numpy')

    # return dict of lists
    if dtype == 'dict':
        pdm_dict = dict()
        for i in range(node_count):
            pdm_dict[i] = list()

    cdef np.double_t dist_ij = 0
    cdef np.int_t i_node
    cdef np.int_t j_node

    for i in range(node_count):
        for j in range(i + 1, node_count):
            if i < leaf_node_count and j >= leaf_node_count and int_r is not None:
                # leaf to internal
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    int_r[j_node],
                    leaf_dir[i],
                    int_dir[j_node],
                    curvature)
            elif i < leaf_node_count and j < leaf_node_count:
                # leaf to leaf
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    leaf_r[j],
                    leaf_dir[i],
                    leaf_dir[j],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance(
                    int_r[i_node],
                    int_r[j_node],
                    int_dir[i_node],
                    int_dir[j_node],
                    curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = np.log(np.cosh(dist_ij))

            if dtype == 'dict':
                pdm_dict[i].append(Cu_edge(dist_ij, i, j))
                pdm_dict[j].append(Cu_edge(dist_ij, j, i))
            elif dtype == 'numpy':
                pdm_np[i, j] = pdm_np[j, i] = dist_ij

    if dtype == 'dict':
        return pdm_dict
    elif dtype == 'numpy':
        return pdm_np