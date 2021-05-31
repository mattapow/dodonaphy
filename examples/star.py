"""
Sample Gaussian points around a perfect star topology in a hyperboloid.
View them on a Poincare disk and make a tree out these samples.
"""

import from .src.hyperboloid as hyp
from from ..src.utils import utilFunc
import torch
from matplotlib import pyplot as plt
import matplotlib.cm
import math

def star():
    S = 6  # n_seqeunces
    DIM = 2  # Corresponds to H^DIM in R^DIM+1.
    HEIGHT = 5  # HEIGHT of points on hyperboloid
    N_SAMPLES = 1  # number of trees to sample

    # Embed points into a star tree at a given HEIGHT from above the "origin" on
    # a hyperboloid. DIM must be 2, i.e. the hyperboloid is a surface in R^3
    assert DIM == 2
    mu = hyp.embed_star_hyperboloid_2d(HEIGHT, S)

    # Variational model on hyperboloid
    # NB: points are in R^n+1 but distribution is on surface with dimension DIM
    STD = 0.3
    VarationalParams = {"sigma": STD*torch.ones(2*S-2, DIM),
                        "xyz_mu": mu}

    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(1, 1, 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')

    # Plot projection of means onto Poincare
    for i in range(2*S-2):
        xy_mu = hyp.hyper_to_poincare(mu[i, :])
        plt.plot(xy_mu[0], xy_mu[1], 'ok')

    for i in range(N_SAMPLES):
        # sample the positions from the VM
        sample_xyz = torch.zeros(2*S-2, DIM+1)
        sample_xy = torch.zeros(2*S-2, DIM)
        for j in range(2*S-2):
            # Take a gaussian sample
            sample_xyz[j, :] = hyp.sample_normal_hyper(
                VarationalParams["xyz_mu"][j, :],
                VarationalParams["sigma"][j, :], DIM)

            # convert to poincare ball
            sample_xy[j, :] = hyp.hyper_to_poincare(sample_xyz[j, :])

        # create the corresponding peel
        (leaf_r, int_r, leaf_dir, int_dir) = utilFunc.cart_to_dir(sample_xy)
        sample_peel = utilFunc.make_peel(
            leaf_r, leaf_dir, int_r, int_dir)

        # add fake root to end (already in make_peel)
        root_xy = torch.unsqueeze(sample_xy[0, :], 0)
        sample_xy = torch.cat((sample_xy, root_xy), dim=0)

        # Plot the tree
        utilFunc.plot_tree(ax, sample_peel, sample_xy, cmap(i/N_SAMPLES))


def embed_star_hyperboloid_2d(height=2, nseqs=6):
    """Embed points in 2D hyperboloid H^2 spread evenly around a circle.

    Points are at height z on Hyperboloid H^2 in R^3.

    Parameters
    ----------
    height : Double, optional
        Height of points on the hyperboloid. The default is 2.
    nseqs : Int, optional
        Number of tip points (sequences). The default is 6.

    Returns
    -------
    nseqs x 3 tensor with positions of the points in R^3 on the Hyperboloid.

    """
    assert height > 1
    x = torch.zeros(2 * nseqs - 2, 3)
    x[:-1, 0] = height
    x[-1, 0] = 1  # origin point

    for i in range(nseqs):
        x[i, 1] = math.sqrt(-1 + height ** 2) * math.cos(i * (2 * math.pi / nseqs))
        x[i, 2] = math.sqrt(-1 + height ** 2) * math.sin(i * (2 * math.pi / nseqs))

    return x

def embed_poincare_star(r=.5, nseqs=6):
    """
    Embed equidistant points at radius r in Poincare disk

    Parameters
    ----------
    r : Float, optional
        0<radius<1. The default is .5.
    nseqs : Int, optional
        Number of points/ sequences to embed. The default is 6.

    Returns
    -------
        Tensor
        Positions of points on 2D Poincaré disk.

    """

    assert abs(r) < 1

    x = torch.zeros(nseqs + 1, 1)
    y = torch.zeros(nseqs + 1, 1)
    dists = torch.zeros(nseqs + 1, nseqs + 1)
    for i in range(nseqs):
        x[i] = r * torch.cos(i * (2 * math.pi / nseqs))
        y[i] = r * torch.sin(i * (2 * math.pi / nseqs))

        # distance between equidistant points at radius r
        dists[i][nseqs] = r
        for j in range(i):
            d2_ij = torch.pow((x[i] - x[j]), 2) + torch.pow((y[i] - y[j]), 2)
            d2_i0 = torch.pow(x[i], 2) + torch.pow(y[i], 2)
            d2_j0 = torch.pow(x[j], 2) + torch.pow(y[j], 2)
            dists[i][j] = torch.arccosh(
                1 + 2 * (d2_ij / ((1 - d2_i0) * (1 - d2_j0))))

    dists = dists + dists.transpose(0, 1)

    emm = {'x': x,
           'y': y,
           'D': dists}
    return emm


star()