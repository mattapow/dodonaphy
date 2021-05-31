"""
Sample Gaussian points around a perfect star topology in a hyperboloid.
View them on a Poincare disk and make a tree out these samples.
"""

import dodonaphy.hyperboloid as hyp
from dodonaphy.utils import utilFunc
import torch
from matplotlib import pyplot as plt
import matplotlib.cm

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

star()