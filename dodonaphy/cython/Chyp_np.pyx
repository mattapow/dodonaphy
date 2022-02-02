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
from dodonaphy import Ctransforms
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
        1D vector origin of initial tangent space T_x H^n in R^d+1.
    y : Tensor
        1D tensor origin of target tangent space T_y H^n in R^d+1.

    Returns
    -------
        1D tensor in tangent space at T_y.

    """
    cdef np.double_t alpha = -lorentz_product(x, y)
    cdef np.double_t coef = lorentz_product(y, xi) / (alpha + 1.0)
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


cpdef tangent_to_hyper(np.ndarray[np.double_t, ndim=1] mu, np.ndarray[np.double_t, ndim=1] v_tilde, np.int_t dim):
    """Project a vector onto the hyperboloid

    Project a vector from the origin, v_tilde, in the tangent space at the origin T_0 H^n
    onto the hyperboloid H^n at point mu

    Parameters
    ----------
    mu: Base point of new tangent space on sheet in R^dim+1
    v_tilde: Vector in T_mu H^n = R^dim to project
    dim: dimension of hyperbolic space H^n

    Returns
    -------
    A point in R^dim+1 on the hyperboloid sheet

    """
    cdef np.ndarray[np.double_t, ndim=1] mu0 = np.concatenate(([1.0], np.zeros(dim, dtype=np.double)))
    cdef np.ndarray[np.double_t, ndim=1] v = np.concatenate(([0.0], v_tilde))
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
    cdef np.double_t norm = np.sum(np.power(location[1:], 2))  #TODO check [1:]
    cdef np.double_t det = np.divide(np.power(a, 2) + norm, np.power(a, 2 * (dim + 1)))

    # compute Jacobian matrix then get determinant
    # J = np.zeros((dim-1, dim))
    # for i in range(dim-1):
    #     J[i, 0] = - location[i+1] / (1+location[0])**2
    #     J[i, i+1] = 1 / (1 + location[0])
    # det = np.linalg.det(np.power(np.matmul(J.T, J), .5))
    return np.log(np.abs(det))


cpdef project_up(np.ndarray[np.double_t, ndim=1] loc):
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
    cdef np.int_t dim = len(loc)
    cdef np.ndarray[np.double_t, ndim=1] out = np.empty((dim+1), dtype=np.double)
    cdef np.double_t z = np.sqrt(np.sum(np.power(loc, 2)) + 1)
    out[1:] = loc
    out[0] = z
    return out

cpdef project_up_2d(np.ndarray[np.double_t, ndim=2] loc):
    z = np.expand_dims(np.sqrt(np.sum(np.power(loc, 2), 1) + 1), 1)
    return np.concatenate((z, loc), axis=1)

cpdef hyperbolic_distance(
    np.ndarray[np.double_t, ndim=1] x1,
    np.ndarray[np.double_t, ndim=1]x2,
    np.double_t curvature=-1.0):
    """Generates hyperbolic distance between two points in hyperboloid.
    Compute using Lorentz product.

    Returns:
        tensor: distance between point 1 and point 2
    """
    x1_sheet = project_up(x1)
    x2_sheet = project_up(x2)

    if np.isclose(curvature, 0.0):
        return np.linalg.norm(x2_sheet-x1_sheet)

    cdef np.double_t inner = np.maximum(-lorentz_product(x1_sheet, x2_sheet), 1.+eps)
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

cpdef get_pdm(
    np.ndarray[np.double_t, ndim=2] x,
    np.double_t curvature=-1.0,
    bint matsumoto=False
    ):
    """ Given points in H^dim (not including z coordinate),
    compute their pairwise distance.
    """
    x_sheet = project_up_2d(x)
    cdef np.ndarray[np.double_t, ndim=2] X = x_sheet @ x_sheet.T
    cdef np.ndarray[np.double_t, ndim=1] u_tilde = np.sqrt(np.diagonal(X) + 1)
    cdef np.ndarray[np.double_t, ndim=2] H = X - np.outer(u_tilde, u_tilde)
    H = np.minimum(H, -(1 + eps))
    cdef np.ndarray[np.double_t, ndim=2] D = 1 / np.sqrt(-curvature) * np.arccosh(-H)
    if matsumoto:
        D = np.log(np.cosh(D))
    return D

cpdef poincare_to_hyper(np.ndarray[np.double_t, ndim=1] location):
    """
    Take points in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: tensor
        n_points x dim location of points in poincare ball

    Returns
    -------

    """
    cdef int dim = location.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(dim + 1)
    cdef np.double_t a = np.power(location[:], 2).sum(axis=0)
    out[0] = (1 + a) / (1 - a)
    out[1:] = 2 * location[:] / (1 - a + eps)
    return out

cpdef poincare_to_hyper_2d(np.ndarray[np.double_t, ndim=2] location):
    cdef int dim = location.shape[1]
    cdef np.ndarray[np.double_t, ndim=1] a = np.power(location, 2).sum(axis=-1)
    cdef np.ndarray[np.double_t, ndim=1] out0 = (1 + a) / (1 - a)
    cdef np.ndarray[np.double_t, ndim=2] out1 = 2 * location / (1 - np.expand_dims(a, axis=1) + eps)
    cdef np.ndarray[np.double_t, ndim=2] out = np.concatenate((np.expand_dims(out0, axis=1), out1), axis=1)
    return out