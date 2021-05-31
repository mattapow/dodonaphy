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


def lorentz_product(x, y=None):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : Tensor
        1D tensor of a point on the hyperboloid.
    y : Tensor optional
        1D tensor of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    if y is None:
        y = x
    return -x[0] * y[0] + torch.dot(x[1:], y[1:])


def hyperboloid_dists(loc):
    """ Get hyperbolic distances between points in X.
    Distances start from 0 and are positive, as for a metric.

    Parameters
    ----------
    loc : Tensor
        n_points x dim tensor listing point locations on a hyperboloid.

    Returns
    -------
    dists : Tensor
        n_points x nseqs pairwise lorentzian distances.

    """
    n_points = len(loc)
    dists = torch.zeros(n_points, n_points)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            dists[i][j] = -1-lorentz_product(loc[i].squeeze(0), loc[j].squeeze(0))
    dists = dists + torch.transpose(dists, 0, 1)

    # Diagonals
    for i in range(n_points):
        dists[i][i] = -1-lorentz_product(loc[i].squeeze(0))

    return dists


def parallel_transport(xi, x, y):
    """ Transport a vector from the tangent space at the x
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


def exponential_map(x, v):
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


def p2t0(loc):
    """Convert location on Poincare ball to Euclidean space.

    Use stereographic projection of point onto hyperboloid surface in R^n+1,
    then project onto z=1 plane, the tangent plane at the origin.

    Parameters
    ----------
    loc: Location in Poincare ball, loc in R^n with |loc|<1

    Returns
    -------
    Projection of location into the tangent space T_0H^n, which is R^n

    """
    loc_hyp = poincare_to_hyper(loc)
    if loc.ndim == 1:
        return loc_hyp[1:]
    elif loc.ndim == 2:
        return loc_hyp[:, 1:]


def t02p(x, mu, dim):
    """Transform a vector x in Euclidean space to the Poincare disk.

    Take a vector in the tangent space of a hyperboloid at the origin, project it
    onto a hyperboloid surface then onto the poincare ball. Mu is the mean of the
    distribution in the Euclidean space

    Parameters
    ----------
    x: Position of sample in tangent space at origin
    mu: Mean of distribution in tangent space at origin

    Returns
    -------
    Transformed vector x, from tangent plane to Poincare ball

    """
    dim = int(dim)
    if x.ndim == 1:
        n_loc = int(len(x.squeeze())/dim)
        x = x.reshape(n_loc, dim)
    elif x.ndim == 2:
        n_loc = len(x.squeeze())

    if type(x).__module__ == np.__name__:
        x = torch.from_numpy(x)

    if type(mu).__module__ == np.__name__:
        mu = torch.from_numpy(mu)

    x_poin = torch.zeros((n_loc, dim))
    mu_hyp = up_to_hyper(mu.reshape(n_loc, dim))
    for i in range(n_loc):
        x_hyp = tangent_to_hyper(mu_hyp[i, :], x[i, :], dim)
        x_poin[i, :] = hyper_to_poincare(x_hyp)
    return x_poin


def up_to_hyper(loc):
    """ Project directly up onto the hyperboloid

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


def tangent_to_hyper(mu, v_tilde, dim):
    """ Project a vector onto the hyperboloid

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


def hyper_to_poincare(location):
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
    # out = torch.zeros(dim)
    # for i in range(dim):
    #     out[i] = location[i + 1] / (1 + location[0])
    return torch.as_tensor([location[i + 1] / (1 + location[0]) for i in range(dim)])


def poincare_to_hyper(location):
    """
    Take points in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: Tensor
        n_points x dim location of points in poincare ball

    Returns
    -------

    """
    eps = torch.finfo(torch.double).eps
    if location.ndim == 1:
        dim = len(location)
        out = torch.zeros(dim + 1).double()
        a = location[:].pow(2).sum(0)
        out[0] = (1 + a) / (1 - a)
        out[1:] = 2 * location[:] / (1 - a + eps)

    elif location.ndim == 2:
        n_points = location.shape[0]
        dim = location.shape[1]
        a = torch.as_tensor(
            [location[i, :].pow(2).sum(0) for i in range(n_points)])
        out0 = torch.div((1 + a), (1 - a))
        out1 = torch.stack(
            [2 * location[i, :] / (1 - location[i, :].pow(2).sum(0) + eps) for i in range(n_points)])
        out = torch.cat((out0.unsqueeze(dim=1), out1), dim=1)
    return out
