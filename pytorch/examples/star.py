"""
Sample Gaussian points around a perfect star topology in a hyperboloid.
View them on a Poincare disk and make a tree out these samples.
"""

import dodonaphy.hyperboloid as hyp
# from dodonaphy.utils import make_peel
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import torch


def main():
    # Embed points into a star tree at a given height from above the "origin"
    n_seqs = 6
    dim = 2  # Corresponds to H^n in R^n+1.
    height = 5
    X = hyp.embed_star_hyperboloid_2d(height, n_seqs)  # dim must be 2
    n_points = X.shape[0]
    # D = hyp.hyperboloid_dists(X)

    # Variational model on hyperboloid
    # NB: points are in R^n+1 but distribution is on surface with dimension dim
    std = 0.1
    vm = {"sd": std*torch.ones(n_points, dim),
          "mu": X}

    # Sample from the VM
    n_samples = 2
    sample_poincare, mu_poincare = sample(vm, n_samples)
    plot_poincare(sample_poincare, mu_poincare)


def sample(vm, n_samples=1):
    # sample from variational model in H^n
    # returns points on poincare ball in R^n+1
    n_points = vm['sd'].shape[0]
    dim = vm['sd'].shape[1]
    mu_poincare = torch.zeros(n_points, dim)
    sample_poincare = torch.zeros(n_samples, n_points, dim)

    for s in range(n_samples):
        hyper_samples = torch.zeros(n_points, dim+1)
        for i in range(n_points):
            # take a gaussian sample
            hyper_samples[i, :] = hyp.sample_normal_hyper(
                vm["mu"][i, :], vm["sd"][i, :], dim)

            # convert to poincare ball
            sample_poincare[s, i, :] = hyp.hyper_to_poincare(
                hyper_samples[i, :])
            mu_poincare[i, :] = hyp.hyper_to_poincare(vm['mu'][i, :])

    return (sample_poincare, mu_poincare)


def plot_poincare(sample_poincare, mu_poincare):
    # Plot points
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    circ = Circle((0, 0), radius=1, fill=False, edgecolor='k')
    ax.add_patch(circ)

    n_samples = sample_poincare.shape[0]
    for s in range(n_samples):
        plt.plot(sample_poincare[s, :, 0],
                 sample_poincare[s, :, 1], '.', markersize=5)

    plt.plot(torch.Tensor([float('NaN')]), torch.Tensor([float('NaN')]),
             '.', markersize=5, label=r'Sample')
    plt.plot(torch.Tensor([float('NaN')]), torch.Tensor([float('NaN')]),
             'ok', label=r'$\mu$')
    plt.legend()
    plt.plot(mu_poincare[:, 0], mu_poincare[:, 1], 'ok')


def make_trees(sample_poincare):
    # convert cartesion coordinates in R^2 to polar coordinates in R^2
    n_seqs = sample_poincare.shape[1]
    leaf_r = (sample_poincare[:n_seqs, 0]**2 +
              sample_poincare[:n_seqs, 1]**2)**.5
    int_r = (sample_poincare[n_seqs:, 0]**2 +
             sample_poincare[n_seqs, 1]**2)**.5
    leaf_dir = torch.atan2(
        sample_poincare[:n_seqs, 1],  sample_poincare[:n_seqs, 0])
    int_dir = torch.atan2(
        sample_poincare[n_seqs:, 1], sample_poincare[n_seqs:, 0])

    # make peel
    sample_peel = make_peel(leaf_r, leaf_dir, int_r, int_dir)


def plot_tree(peel, X):
    # plot the tree encoded by peel with positions X = [x, y]
    n_parents = peel.shape[0]
    for i in range(n_parents):
        child = peel[i, 0]
        sibling = peel[i, 1]
        parent = peel[i, 2]
        Line2D([X[child, 0], X[parent, 0]], [X[child, 1], X[parent, 1]])
        Line2D([X[sibling, 0], X[parent, 0]], [X[sibling, 1], X[parent, 1]])


if __name__ == '__main__':
    main()
