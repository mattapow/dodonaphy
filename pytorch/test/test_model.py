import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy import DnaCharacterMatrix
from dodonaphy.model import DodonaphyModel
from dodonaphy.phylo import compress_alignment, JC69_p_t, calculate_treelikelihood
from dodonaphy.utils import utilFunc
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from dendropy.interop import raxml


def test_model_init_rand():
    dim = 3    # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # testing raxml
    # rx = raxml.RaxmlRunner(raxml_path="raxmlHPC-AVX2")
    # # tree = rx.estimate_tree(char_matrix=dna, raxml_args=['-e', 'likelihoodEpsilon', '-h' '--JC69'])
    # tree = rx.estimate_tree(char_matrix=dna, raxml_args=["-h", "--JC69"])

    rx = raxml.RaxmlRunner()
    rxml_tree = rx.estimate_tree(char_matrix=dna)
    assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
    dist = np.array([[assemblage_data[i][j] for j in sorted(
        assemblage_data[i])] for i in sorted(assemblage_data)])
    emm = utilFunc.hydra(D=dist, dim=dim)

    partials, weights = compress_alignment(dna)

    # have to create and convert dist into hyperbolic embedding
    simtree.print_plot()

    mymod = DodonaphyModel(partials, weights, dim)
    mymod.learn(epochs=100)

    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)
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
    seqlen = 1000  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # Compute RAxML tree likelihood
    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    peel, blens = utilFunc.dendrophy_to_pb(tree)
    mats = JC69_p_t(blens)
    rml_L = calculate_treelikelihood(partials, weights, peel, mats,
                                     torch.full([4], 0.25, dtype=torch.float64))
    print("RAxML Likelihood: " + str(rml_L.item()))
    print("NB: ELBO is: Likelihood - log(Q) + Jacobian(=1) + logPrior(=0)")
    # TODO: compare sample likelihood instead of elbos?

    # Get tip distances
    pdm = simtree.phylogenetic_distance_matrix()
    dists = np.zeros((S, S))
    for i, t1 in enumerate(simtree.taxon_namespace):
        for j, t2 in enumerate(simtree.taxon_namespace):
            dists[i][j] = pdm(t1, t2)
            dists[j][i] = pdm(t1, t2)

    # embed tips with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
        emm["r"]), torch.from_numpy(emm["directional"]))

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes to narrow distributions at origin
    param_init = {
        "leaf_x_mu": torch.zeros_like(leaf_loc_poin, requires_grad=True, dtype=torch.float64),
        "leaf_x_sigma": torch.full([S], 1/50, requires_grad=True, dtype=torch.float64),
        "int_x_mu": torch.zeros(S-2, dim, requires_grad=True, dtype=torch.float64),
        "int_x_sigma": torch.full([S-2], 1/50, requires_grad=True, dtype=torch.float64)
    }

    # Plot initial embedding if dim==2
    mymod.learn(param_init=param_init, epochs=0)
    nsamples = 3
    if dim == 2:
        plt.figure(figsize=(7, 7), dpi=300)
        fig, ax = plt.subplots(1, 2)
        peels, blens, X = mymod.draw_sample(nsamples)
        ax[0].set(xlim=[-1, 1])
        ax[0].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax[0], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[0].set_title("Original Embedding Sample")

    # learn
    mymod.learn(param_init=param_init, epochs=1000)
    peels, blens, X = mymod.draw_sample(nsamples)

    # draw the tree samples if dim==2
    if dim == 2:
        ax[1].set(xlim=[-1, 1])
        ax[1].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax[1], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[1].set_title("Final Embedding Sample")
        fig.show()