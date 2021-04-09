import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy import DnaCharacterMatrix
from dodonaphy.model import DodonaphyModel
from dodonaphy.phylo import compress_alignment, JC69_p_t, calculate_treelikelihood
from dodonaphy.utils import utilFunc
from dodonaphy.hyperboloid import p2t0
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from dendropy.interop import raxml
from dendropy import treecalc
import pytest

def test_model_init_rand():
    dim = 3    # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    s = simtree.postorder_node_iter()
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

    # try a tree construction from peel and blens data
    rxml_tree.print_plot()

    tip_labels = simtree.taxon_namespace.labels()
    rxml_peel, rxml_blens = utilFunc.dendrophy_to_pb(rxml_tree)
    rxml_tree_nw = utilFunc.tree_to_newick(tip_labels, rxml_peel, rxml_blens)
    rxml_peel_dp = dendropy.Tree.get(data=rxml_tree_nw, schema="newick")
    dodonaphy_tree_nw = utilFunc.tree_to_newick(simtree.taxon_namespace.labels(), peels[0], blens[0])
    dodonaphy_tree_dp = dendropy.Tree.get(data=dodonaphy_tree_nw, schema="newick")
    # dodonaphy_tree_dp = dendropy.TreeList(taxon_namespace=rxml_tree.taxon_namespace)

    
    # compare raxml and dodonaphy tree based on euclidean and Robinson_foulds distance
    ec_dist = treecalc.euclidean_distance(rxml_peel_dp, dodonaphy_tree_dp)
    # rf_dist = treecalc.robinson_foulds_distance(rxml_tree, dodonaphy_tree_dp)


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
    S = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    # TODO: Partials is list, which is sometimes an issue for calculate_treelikelihood, expects tensor
    partials, weights = compress_alignment(dna)
    # mymod = DodonaphyModel(partials, weights, dim)
    DodonaphyModel(partials, weights, dim)

    # Compute RAxML tree likelihood
    # TODO: set RAxML to use --JC69. Confirm in log file
    print('Warning: RAxML using GTR model not JC69')
    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    peel, blens = utilFunc.dendrophy_to_pb(tree)
    mats = JC69_p_t(blens)
    rml_L = calculate_treelikelihood(partials, weights, peel, mats,
                                     torch.full([4], 0.25, dtype=torch.float64))
    print("RAxML Likelihood: " + str(rml_L.item()))
    print("NB: ELBO is: Likelihood - log(Q) + Jacobian + logPrior(=0)")

    # Get tip distances
    pdm = simtree.phylogenetic_distance_matrix()
    dists = np.zeros((S, S))
    for i, t1 in enumerate(simtree.taxon_namespace):
        for j, t2 in enumerate(simtree.taxon_namespace):
            dists[i][j] = pdm(t1, t2)
            dists[j][i] = pdm(t1, t2)

    # embed tips with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))
    leaf_loc_t0 = p2t0(leaf_loc_poin)

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes to narrow distributions at origin
    cv = 1./50
    leaf_sigma = np.abs(np.array(leaf_loc_t0)) * cv
    param_init = {
        "leaf_x_mu": leaf_loc_t0.requires_grad_(True),
        "leaf_x_sigma": torch.tensor(leaf_sigma,requires_grad=True, dtype=torch.float64),
        "int_x_mu": torch.zeros(S-2, dim, requires_grad=True, dtype=torch.float64),
        "int_x_sigma": torch.full([S-2], 1/100, requires_grad=True, dtype=torch.float64)
    }

    # Plot initial embedding if dim==2
    mymod.learn(param_init=param_init, epochs=0)
    nsamples = 1
    if dim == 2:
        plt.figure(figsize=(7, 7), dpi=600)
        fig, ax = plt.subplots(1, 2)
        peels, blens, X = mymod.draw_sample(nsamples)
        ax[0].set(xlim=[-1, 1])
        ax[0].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax[0], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[0].set_title("Original Embedding Sample")
        plt.close()

    # learn
    mymod.learn(param_init=param_init, epochs=100)
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


def test_calculate_likelihood():
    """
    Sometimes calculate_likelihood was throwing errors about
    torch.matmul(Tensor, list), where it wanted torch.matmul(Tensor, Tensor)
    """

    dim = 1  # number of dimensions for embedding
    S = 4  # number of sequences to simulate
    seqlen = 10  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # Compute RAxML tree
    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    peel, blens = utilFunc.dendrophy_to_pb(tree)
    mats = JC69_p_t(blens)

    _ = calculate_treelikelihood(partials, weights, peel, mats,
                                     torch.full([4], 0.25, dtype=torch.float64))

