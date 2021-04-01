import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy import DnaCharacterMatrix
from dodonaphy.model import DodonaphyModel
from dodonaphy.phylo import compress_alignment
from dodonaphy.utils import utilFunc
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch


def testFunc():
    dim = 3    # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    partials, weights = compress_alignment(dna)

    # print(weights)
    # have to create and convert dist into hyperbolic embedding
    simtree.print_plot()

    mymod = DodonaphyModel(partials, weights, dim)
    mymod.learn(epochs=10)

    nsamples = 3
    peels, blens, X = mymod.draw_sample(nsamples)
    # maximum likelihood parameter values

    # draw the tree samples
    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(1, 1, 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(nsamples):
        utilFunc.plot_tree(
            ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
    plt.show()


def test_model_init_hydra():
    """
    Initialise embedding with hydra
    Optimise VM
    Plot samples from VM
    """
    dim = 2  # number of dimensions for embedding
    S = 5  # number of sequences to simulate
    seqlen = 100  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    partials, weights = compress_alignment(dna)

    # get tip distances
    pdm = simtree.phylogenetic_distance_matrix()
    dists = np.zeros((S, S))
    for i, t1 in enumerate(simtree.taxon_namespace):
        for j, t2 in enumerate(simtree.taxon_namespace):
            dists[i][j] = pdm(t1, t2)
            dists[j][i] = pdm(t1, t2)

    # embed tips with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes to narrow distributions at origin
    param_init = {
        "leaf_x_mu": leaf_loc_poin.double(),
        "leaf_x_sigma": torch.ones(S).double()/50,
        "int_x_mu": torch.zeros(S-2, dim).double(),
        "int_x_sigma": torch.ones(S-2).double()/50
    }

    mymod = DodonaphyModel(partials, weights, dim)
    mymod.learn(param_init=param_init, epochs=0)
    nsamples = 3

    if dim == 2:
        # Plot initial embedding
        plt.figure(figsize=(7, 7), dpi=300)
        fig, ax = plt.subplots(1, 2)
        peels, blens, X = mymod.draw_sample(nsamples)
        ax[0].set(xlim=[-1, 1])
        ax[0].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(ax[0], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[0].set_title("Original distribution")
        plt.show()

    # learn
    mymod.learn(param_init=param_init, epochs=10)
    peels, blens, X = mymod.draw_sample(nsamples)

    if dim == 2:
        # draw the tree samples
        ax[1].set(xlim=[-1, 1])
        ax[1].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(ax[1], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[1].set_title("Final distribution")
        fig.show()
