# Some functions from https://github.com/HazyResearch/HypHC
import torch
from . import hyperboloid


def isometric_transform(a, x):
    """Reflection (circle inversion of x through orthogonal circle centered at a)."""
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
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
    """ computes the point on the geodesic segment from o to x at half the distance """
    return x / (1. + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))


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


def incentre(a, b, return_coord=True):
    """
    Copmute incentre of the triangle formed by o, a and b
    """
    # TODO: use hyperbolic centre. Possibly in https://arxiv.org/pdf/1410.6735.pdf
    w_a = hyp_dist_o(b)
    w_b = hyp_dist_o(a)
    ab = torch.stack((a, b))
    w_c = hyperboloid.hyperboloid_dists(hyperboloid.poincare_to_hyper(ab))[0][1]
    proj = (w_a * a + w_b * b) / (w_a + w_b + w_c)

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


# def get_theta(x, y):
#     """
#     Computes the angle between two vectors
#     """
#     return torch.acos(torch.dot(x, y) / torch.norm(x) / torch.norm(y))


# def get_alpha(x, y):
#     """Angle between x and y
#     Using eq. 8 from https://arxiv.org/abs/2010.00402

#     Args:
#         x (tensor): first point
#         y (tensor): second point

#     Returns:
#         Scalar tensor: Angle between x and y
#     """
#     theta = get_theta(x, y)
#     x_norm = x.norm(dim=-1, p=2, keepdim=True)
#     y_norm = y.norm(dim=-1, p=2, keepdim=True)

#     alpha = (x_norm * (y_norm)**2 + 1) / (y_norm * (x_norm)**2 + 1) - torch.cosh(theta)
#     return torch.atan(alpha / torch.sin(theta))


# def min_norm(a, b):
#     """Return whichever input has the minimum norm

#     Args:
#         a (tensor): First tensor
#         b (tensor): Second tensor
#     """
#     if torch.norm(a) < torch.norm(b):
#         return a
#     return b
