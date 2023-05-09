""" Lorentz model of hyperbolic space

Using a hyperboloid surface H^n sitting in R^n+1. Points x in R^n+1 with
lorentz product <x,x>=-1 are on the hyperboloid surface.

If n=2, then we can stereo-graphically project the surface into the Poincare
disk as they are isomorphic.

Generally takes torch tensors as inputs.

General methodology coming from Nagano 2019
A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning
https://github.com/pfnet-research/hyperbolic_wrapped_distribution
"""

import torch
import numpy as np
cimport numpy as np
from collections import defaultdict
from dodonaphy import Ctransforms
from dodonaphy.edge import Edge

eps = torch.tensor(torch.finfo(torch.double).eps)

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

cpdef parallel_transport(xi, x, y):
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
    alpha = -lorentz_product(x, y)
    coef = lorentz_product(y, xi) / (alpha + 1)
    return xi + coef * (x + y)


cpdef exponential_map(x, v):
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
    vnorm = torch.sqrt(torch.clamp(lorentz_product(v), min=eps))
    return torch.cosh(vnorm) * x + torch.sinh(vnorm) * v / vnorm


cpdef exp_map_inverse(z, mu):
    """Inverse of exponential map

    Args:
        z ([type]): Hyperboloid location in R^n+1
        mu ([type]): Tangent point

    Returns:
        [type]: [description]
    """
    alpha = -lorentz_product(mu, z)
    factor = torch.acosh(alpha) / (torch.sqrt(alpha ** 2 - 1))
    return factor * (z - alpha * mu)


cpdef hyperbolic_distance(x1, x2, curvature=-torch.ones(1)):
    """Generates hyperbolic distance between two points in poincare ball.

    Args:
        x1 (1D tensor): directional of point 1
        x2 (1D tensor): directional of point 2
        curvature (tensor): curvature

    Returns:
        tensor: distance between point 1 and point 2
    """

    if abs(curvature + 1.) > .000000001:
        return hyperbolic_distance_lorentz(x1, x2, curvature)

    invariant = 2 * torch.sum((x2-x1)**2) / (1-torch.linalg.norm(x1)**2) / (1-torch.linalg.norm(x2)**2)
    if torch.isnan(invariant):
        return hyperbolic_distance_lorentz(x1, x2, curvature)
    return torch.acosh(1 + invariant)


cpdef hyperbolic_distance_lorentz(x1, x2, curvature=-torch.ones(1)):
    """Generates hyperbolic distance between two points in poincare ball.
    Project onto hyperboloid and compute using Lorentz product.

    Returns:
        tensor: distance between point 1 and point 2
    """

    if torch.isclose(curvature, torch.zeros(1)):
        return torch.norm(x2-x1)

    z1 = poincare_to_hyper(x1).squeeze()
    z2 = poincare_to_hyper(x2).squeeze()
    inner = torch.clamp(-lorentz_product(z1, z2), min=1.+eps)
    return 1. / torch.sqrt(-curvature) * torch.acosh(inner)


cpdef lorentz_product(x, y=None):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : tensor
        1D array of a point on the hyperboloid.
    y : tensor optional
        1D array of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    if y is None:
        y = x
    if type(x).__module__ == 'torch':
        return -x[0] * y[0] + torch.dot(x[1:], y[1:])
    elif type(x).__module__ == 'numpy':
        return -x[0] * y[0] + np.dot(x[1:], y[1:])
    raise TypeError('x must be numpy or torch')
    
cpdef poincare_to_hyper(location):
    """
    Take points in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: tensor
        n_points x dim location of points in poincare ball

    Returns
    -------

    """
    cdef int dim
    if location.ndim == 1:
        dim = location.shape[0]
        out = torch.zeros(dim + 1, dtype=torch.double)
        a = location[:].pow(2).sum(dim=0)
        out[0] = (1 + a) / (1 - a)
        out[1:] = 2 * location[:] / (1 - a + eps)
    elif location.ndim == 2:
        dim = location.shape[1]
        a = location.pow(2).sum(dim=-1)
        out0 = torch.div((1 + a), (1 - a))
        out1 = 2 * location / (1 - a.unsqueeze(dim=1) + eps)
        out = torch.cat((out0.unsqueeze(dim=1), out1), dim=1)
    return out

cpdef ball2real(loc_ball, radius=1):
    """A map from the unit ball B^n to real R^n.
    Inverse of real2ball.

    Args:
        loc_ball (tensor): [description]
        radius (tensor): [description]

    Returns:
        tensor: [description]
    """
    if loc_ball.ndim == 1:
        loc_ball = loc_ball.unsqueeze(dim=-1)
    dim = loc_ball.shape[1]
    norm_loc_ball = torch.norm(loc_ball, dim=-1, keepdim=True).repeat(1, dim)
    loc_real = loc_ball / (radius - norm_loc_ball)
    return loc_real


cpdef real2ball(loc_real, radius=1):
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



cpdef real2ball_LADJ(y, radius=1.0):
    """Copmute log of absolute value of determinate of jacobian of real2ball on point y
    Args:
        y (tensor): Points in R^n n_points x n_dimensions
    Returns:
        scalar tensor: log absolute determinate of Jacobian
    """
    if y.ndim == 1:
        y = y.unsqueeze(dim=-1)
    n, D = y.shape
    log_abs_det_J = torch.zeros(1)
    norm = torch.norm(y, dim=-1, keepdim=True)
    for k in range(n):
        J = (torch.eye(D, D) - torch.outer(y[k], y[k]) / (norm[k] * (norm[k] + 1))) / (1+norm[k])
        log_abs_det_J = log_abs_det_J + torch.logdet(radius * J)
    return log_abs_det_J

cpdef t02p(x, mu=None, get_jacobian=False):
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
        mu = torch.zeros_like(x)
    cdef x_poin = torch.zeros_like(x)
    cdef mu_hyp = project_up_2d(mu)
    cdef jacobian = torch.zeros(1)
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


cpdef hyper_to_poincare_jacobian(location):
    cdef int dim = location.shape[0]

    cdef a = 1 + location[0]
    cdef norm = torch.sum(torch.pow(location[1:], 2))
    det = torch.div(torch.pow(a, 2) + norm, torch.pow(a, 2 * (dim + 1)))

    # compute Jacobian matrix then get determinant
    # J = np.zeros((dim-1, dim))
    # for i in range(dim-1):
    #     J[i, 0] = - location[i+1] / (1+location[0])**2
    #     J[i, i+1] = 1 / (1 + location[0])
    # det = np.linalg.det(np.power(np.matmul(J.T, J), .5))
    return torch.log(torch.abs(det))

cpdef tangent_to_hyper(mu, v_tilde, dim):
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
    cdef mu0 = torch.cat((torch.ones(1), torch.zeros(dim, dtype=torch.double)))
    cdef v = torch.cat((torch.ones(1), torch.squeeze(v_tilde)))
    cdef u = parallel_transport(v, mu0, mu)
    cdef z = exponential_map(mu, u)
    return z


cpdef tangent_to_hyper_jacobian(mu, v_tilde, dim):
    """Return log absolute of determinate of jacobian for tangent_to_hyper

    Args:
        mu ([type]): [description]
        v_tilde ([type]): [description]
        dim ([type]): [description]

    Returns:
        Scalar tensor: [description]
    """
    cdef r = torch.norm(v_tilde)
    return torch.log(torch.pow(torch.div(torch.sinh(r), r), dim - 1))


cpdef hyper_to_poincare(location):
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
    cdef int dim = location.shape[0] - 1
    cdef  out = torch.zeros(dim)
    for i in range(dim):
        out[i] = location[i + 1] / (1 + location[0])
    return out


cdef tangent_to_hyper_2d(loc_t0):
    (n_taxa, dim) = loc_t0.shape
    cdef zero_hyp = project_up(torch.zeros((dim), dtype=torch.double))
    cdef x_hyp = torch.zeros((n_taxa, dim+1), dtype=torch.double)
    for i in range(n_taxa):
        x_hyp[i, :] = tangent_to_hyper(zero_hyp, loc_t0[i, :], dim)
    return x_hyp
    

cpdef get_pdm(leaf_loc, curvature=-torch.ones(1), bint matsumoto=False, projection="up"):
    """Get pair-wise hyperbolic distance matrix.


    Args:
        leaf_loc (tensor): leaf locations in H^dim
        curvature (double): curvature. Must be negative.
        matsumoto (bool): apply matsumoto transform

    Returns:
        ndarray: pairwise distance between point
    """
    if projection == "up":
        x_sheet = project_up_2d(leaf_loc)
    elif projection == "wrap":
        x_sheet = tangent_to_hyper_2d(leaf_loc)
    cdef X = x_sheet @ x_sheet.T
    cdef u_tilde = torch.sqrt(torch.diagonal(X) + 1)
    cdef H = X - torch.outer(u_tilde, u_tilde)
    H = torch.minimum(H, -torch.tensor(1.0000001))
    cdef D = 1 / torch.sqrt(-curvature) * torch.arccosh(-H)
    if matsumoto:
        D = torch.log(torch.cosh(D))
    D.fill_diagonal_(0.0)
    return D

cpdef p2t0(loc, mu=None, get_jacobian=False):
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
    cdef vec_hyp = poincare_to_hyper_2d(loc)
    if mu is None:
        mu = torch.zeros_like(loc)
    mu = project_up_2d(mu)
    n_loc, dim = loc.shape
    cdef zero = torch.zeros((dim + 1), dtype=torch.double)
    zero[0] = 1

    cdef out = torch.zeros_like(loc)
    for i in range(n_loc):
        vec_t0_mu = exp_map_inverse(vec_hyp[i, :], mu[i, :])
        vec_t0_0 = parallel_transport(vec_t0_mu, mu[i, :], zero)
        out[i, :] = vec_t0_0[1:]

    if get_jacobian:
        _, jacobian = t02p(out, mu[:, 1:], get_jacobian=True)
        return out, -jacobian
    return out

cpdef poincare_to_hyper_2d(location):
    cdef int dim = location.shape[1]
    cdef a = torch.pow(location, 2).sum(dim=-1)
    cdef out0 = (1 + a) / (1 - a)
    cdef out1 = 2 * location / (1 - a.unsqueeze(dim=1) + eps)
    cdef out = torch.cat((out0.unsqueeze(dim=1), out1), dim=1)
    return out

cpdef project_up(loc):
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
    z = torch.sqrt(torch.sum(torch.pow(loc.clone(), 2), -1, keepdim=True) + 1)
    return torch.cat((z, loc), dim=-1)

cpdef project_up_2d(loc):
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
        return project_up(loc).unsqueeze(-1)
    z = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(loc.clone(), 2), 1) + 1), 1)
    return torch.cat((z, loc), axis=1)