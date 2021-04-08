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
from dendropy.interop import raxml
from dendropy import treecalc

def testFunc():
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
    
    dodonaphy_tree_nw = utilFunc.tree_to_newick(simtree.taxon_namespace.labels(), peels[0], blens[0])
    dodonaphy_tree_dp = dendropy.Tree.get(data=dodonaphy_tree_nw, schema="newick")
    dodonaphy_tree_dp = dendropy.TreeList(taxon_namespace=rxml_tree.taxon_namespace)
    # maximum likelihood parameter values

    # compare raxml and dodonaphy tree based on euclidean and Robinson_foulds distance
    ec_dist = treecalc.euclidean_distance(rxml_tree, dodonaphy_tree_dp)
    rf_dist = treecalc.robinson_foulds_distance(rxml_tree, dodonaphy_tree_dp)


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

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=data, raxml_args=["--no-bfgs"])
    tree.plot_tree()

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

    mymod = DodonaphyModel(partials, weights, dim)
    mymod.learn(param_init=param_init, epochs=1000)
    # mymod.learn(epochs=1000)
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
            utilFunc.plot_tree(
                ax[0], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[0].set_title("Original distribution")
        plt.show()

        # learn
        mymod.learn(param_init=param_init, epochs=1000)
        peels, blens, X = mymod.draw_sample(nsamples)

        if dim == 2:
            # draw the tree samples
            ax[1].set(xlim=[-1, 1])
            ax[1].set(ylim=[-1, 1])
            cmap = matplotlib.cm.get_cmap('Spectral')
            for i in range(nsamples):
                utilFunc.plot_tree(
                    ax[1], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
            ax[1].set_title("Final distribution")
            fig.show()

testFunc()
