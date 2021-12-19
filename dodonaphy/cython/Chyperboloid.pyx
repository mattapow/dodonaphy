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
from dodonaphy import utils, Cutils
from dodonaphy.edge import Edge

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
    eps = torch.finfo(torch.double).eps
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
    if loc.dim() < 2:
        loc = loc.unsqueeze(dim=-1)

    vec_hyp = poincare_to_hyper(loc)
    if mu is None:
        mu = torch.zeros_like(loc)
        mu = up_to_hyper(mu)
    n_loc, dim = loc.shape
    zero = torch.zeros((dim + 1)).double()
    zero[0] = 1

    out = torch.zeros_like(loc)
    for i in range(n_loc):
        vec_t0_mu = exp_map_inverse(vec_hyp[i, :], mu[i, :])
        vec_t0_0 = parallel_transport(vec_t0_mu, mu[i, :], zero)
        out[i, :] = vec_t0_0[1:]

    if get_jacobian:
        _, jacobian = t02p(out, mu[:, 1:], get_jacobian=True)
        return out, -jacobian
    return out


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
    n_loc, dim = x.shape

    if type(x).__module__ == np.__name__:
        x = torch.from_numpy(x)

    if type(mu).__module__ == np.__name__:
        mu = torch.from_numpy(mu)

    if mu is None:
        mu = torch.zeros_like(x)

    x_poin = torch.zeros_like(x)
    mu_hyp = up_to_hyper(mu)
    jacobian = torch.zeros(1)
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
    mu_dbl = mu.double()
    mu0_dbl = torch.cat((torch.ones(1), torch.zeros(dim, dtype=torch.double)))

    v = torch.cat((torch.zeros(1), torch.squeeze(v_tilde))).double()
    u = parallel_transport(v, mu0_dbl, mu_dbl)
    z = exponential_map(mu_dbl, u)
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
    r = torch.norm(v_tilde)
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
    dim = location.shape[0] - 1
    out = torch.zeros(dim)
    for i in range(dim):
        out[i] = location[i + 1] / (1 + location[0])
    return out
    # return torch.as_tensor([location[i + 1] / (1 + location[0]) for i in range(dim)])


cpdef hyper_to_poincare_jacobian(location):
    dim = location.shape[0]

    # precomputed determinant
    a = 1 + location[0]
    norm = torch.sum(torch.pow(location[1:], 2))
    det = torch.div(torch.pow(a, 2) + norm, torch.pow(a, 2 * (dim + 1)))

    # compute Jacobian matrix then get determinant
    # J = torch.zeros((dim-1, dim))
    # for i in range(dim-1):
    #     J[i, 0] = - location[i+1] / (1+location[0])**2
    #     J[i, i+1] = 1 / (1 + location[0])
    # det = torch.det(torch.pow(torch.matmul(J.T, J), .5))
    return torch.log(torch.abs(det))


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
        z = torch.sqrt(torch.sum(torch.pow(loc, 2)) + 1).unsqueeze(0)
        return torch.cat((z, loc), dim=0)
    elif loc.ndim == 2:
        z = torch.sqrt(torch.sum(torch.pow(loc, 2), 1) + 1).unsqueeze(1)
        return torch.cat((z, loc), dim=1)


cpdef hyperbolic_distance(r1, r2, directional1, directional2, curvature=-torch.ones(1)):
    """Generates hyperbolic distance between two points in poincare ball.

    Args:
        r1 (tensor): radius of point 1
        r2 (tensor): radius of point 2
        directional1 (1D tensor): directional of point 1
        directional2 (1D tensor): directional of point 2
        curvature (tensor): curvature

    Returns:
        tensor: distance between point 1 and point 2
    """
    x1 = utils.dir_to_cart(r1, directional1)
    x2 = utils.dir_to_cart(r2, directional2)

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
    eps = torch.finfo(torch.float64).eps
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
    cdef double eps = 0.0000000000000003
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


cpdef get_pdm(
    leaf_r, leaf_dir, int_r=None, int_dir=None, curvature=-torch.ones(1), astorch=False
):
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0]
    if int_r is not None:
        node_count = node_count + int_r.shape[0]

    if astorch:
        pdm = torch.zeros((node_count, node_count)).double()
    else:
        pdm = defaultdict(list)

    for i in range(node_count):
        for j in range(i + 1, node_count):
            dist_ij = 0

            if (i < leaf_node_count) and (j < leaf_node_count):
                # leaf to leaf
                dist_ij = hyperbolic_distance(
                    leaf_r[i], leaf_r[j], leaf_dir[i], leaf_dir[j], curvature
                )
            elif i < leaf_node_count:
                # leaf to internal
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    int_r[j - leaf_node_count],
                    leaf_dir[i],
                    int_dir[j - leaf_node_count],
                    curvature,
                )
            else:
                # internal to internal
                i_node = i - leaf_node_count
                dist_ij = hyperbolic_distance(
                    int_r[i_node],
                    int_r[j - leaf_node_count],
                    int_dir[i_node],
                    int_dir[j - leaf_node_count],
                    curvature,
                )

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = torch.log(torch.cosh(dist_ij))

            if astorch:
                pdm[i, j] = pdm[j, i] = dist_ij
            else:
                pdm[i].append(Edge(dist_ij, i, j))
                pdm[j].append(Edge(dist_ij, j, i))

    return pdm



cpdef get_pdm_torch(leaf_r, leaf_dir, int_r=None, int_dir=None, curvature=-torch.ones(1)):
    """Pair-wise hyperbolic distance matrix

        Note if curvature=0, then the SQUARED Euclidean distance is computed.
    Args:
        leaf_r (tensor):
        leaf_dir (tensor):
        int_r (1D tensor):
        inr_dir (1D tensor):
        curvature (double): curvature

    Returns:
        ndarray: pairwise distance between point
    """
    cdef int leaf_node_count = leaf_r.shape[0]
    cdef int int_node_count = 0
    if int_r is None:
        int_r = torch.tensor((0)).unsqueeze(dim=-1)
        int_dir = torch.tensor((0, 0)).unsqueeze(dim=-1)
    else:
        int_node_count = int_r.shape[0]
    cdef int node_count = leaf_node_count + int_node_count
    
    cdef pdm = torch.zeros((node_count, node_count)).double()

    cdef int i_node
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
            dist_ij = torch.log(torch.cosh(dist_ij))

            pdm[i, j] = pdm[j, i] = dist_ij

    return pdm