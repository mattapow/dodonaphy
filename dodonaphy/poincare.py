# Some functions from https://github.com/HazyResearch/HypHC
import numpy as np
import torch


def isometric_transform(a, x):
    """Reflection (circle inversion of x through orthogonal circle centered at a)."""
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.0
    u = x - a
    return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a


def reflection_center(mu):
    """Center of inversion circle."""
    return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)


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


def _halve(x):
    """computes the point on the geodesic segment from o to x at half the distance"""
    return x / (1.0 + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))


def hyp_lca(a, b, return_coord=True):
    """
    Computes projection of the origin on the geodesic between a and b, at scale c
    More optimized than hyp_lca1
    """
    if torch.allclose(a, b):
        proj = a.clone()

    r = reflection_center(a)
    b_inv = isometric_transform(r, b)
    o_inv = a
    o_inv_ref = euc_reflection(o_inv, b_inv)
    o_ref = isometric_transform(r, o_inv_ref)
    proj = _halve(o_ref)

    if not return_coord:
        return hyp_dist_o(proj)
    else:
        return proj


def hyp_dist_o(x):
    """
    Computes hyperbolic distance between x and the origin.
    """
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    return 2 * torch.arctanh(x_norm)


def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)
