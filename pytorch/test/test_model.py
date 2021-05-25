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
from ete3 import Tree


def test_model_init_rand():
    """Testing Dodonaphy model with randomly initialized parameters for variational inference
    """
    dim = 3    # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # testing raxml
    # rx = raxml.RaxmlRunner(raxml_path="raxmlHPC-AVX2")
    # # tree = rx.estimate_tree(char_matrix=dna, raxml_args=['-e', 'likelihoodEpsilon', '-h' '--JC69'])
    # tree = rx.estimate_tree(char_matrix=dna, raxml_args=["-h", "--JC69"])

    # rx = raxml.RaxmlRunner()
    # rxml_tree = rx.estimate_tree(char_matrix=dna)
    # assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
    # dist = np.array([[assemblage_data[i][j] for j in sorted(
    #     assemblage_data[i])] for i in sorted(assemblage_data)])
    # emm = utilFunc.hydra(D=dist, dim=dim)

    # model initiation and training
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)
    # variational parameters: [default] randomly generated within model constructor
    mymod.learn(epochs=10)

    # draw samples from variational posterior
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

    # compare dodonapy with RAxML
    # tip_labels = simtree.taxon_namespace.labels()
    # rxml_peel, rxml_blens = utilFunc.dendrophy_to_pb(rxml_tree)
    # rxml_tree_nw = utilFunc.tree_to_newick(tip_labels, rxml_peel, rxml_blens)
    # rxml_peel_dp = dendropy.Tree.get(data=rxml_tree_nw, schema="newick")
    dodonaphy_tree_nw = utilFunc.tree_to_newick(
        simtree.taxon_namespace.labels(), peels[0], blens[0])
    dodonaphy_tree_dp = dendropy.Tree.get(
        data=dodonaphy_tree_nw, schema="newick")
    dodonaphy_tree_dp.print_plot()
    # dodonaphy_tree_dp = dendropy.TreeList(taxon_namespace=rxml_tree.taxon_namespace)

    # # compare raxml and dodonaphy tree based on euclidean and Robinson_foulds distance
    # ec_dist = treecalc.euclidean_distance(rxml_peel_dp, dodonaphy_tree_dp)
    # rf_dist = treecalc.robinson_foulds_distance(rxml_tree, dodonaphy_tree_dp)

    # draw the tree samples
    if dim == 2:
        plt.figure(figsize=(7, 7), dpi=100)
        ax = plt.subplot(1, 1, 1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        plt.show()


def test_draws_different():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # learn
    mymod.learn(epochs=0)

    # draw
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)
    assert not torch.equal(blens[0], blens[1])
    assert not torch.equal(blens[0], blens[2])
    assert not torch.equal(blens[1], blens[2])

    assert not torch.equal(lp__[0], lp__[1])
    assert not torch.equal(lp__[0], lp__[2])
    assert not torch.equal(lp__[1], lp__[2])


def test_init_RAxML_hydra():
    """
    Initialise the emebedding with RAxML distances given to hydra.
    RAxML gives internal nodes as well.
    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # s = simtree.postorder_node_iter()
    # testing raxml
    # rx = raxml.RaxmlRunner(raxml_path="raxmlHPC-AVX2")
    # # tree = rx.estimate_tree(char_matrix=dna, raxml_args=['-e', 'likelihoodEpsilon', '-h' '--JC69'])
    # tree = rx.estimate_tree(char_matrix=dna, raxml_args=["-h", "--JC69"])

    rx = raxml.RaxmlRunner()
    rxml_tree = rx.estimate_tree(char_matrix=dna)
    assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
    dist = np.array([[assemblage_data[i][j] for j in sorted(
        assemblage_data[i])] for i in sorted(assemblage_data)])
    emm = utilFunc.hydra(D=dist, dim=dim, equi_adj=0.)

    leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
        emm["r"]), torch.from_numpy(emm["directional"]))
    leaf_loc_t0 = p2t0(leaf_loc_poin).detach().numpy()

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes to narrow distributions at origin
    cv = 1. / 50
    eps = np.finfo(np.double).eps
    leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
    param_init = {
        "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
        "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
        "int_mu": torch.zeros(nseqs - 2, dim, requires_grad=True, dtype=torch.float64),
        "int_sigma": torch.full((nseqs - 2, dim), np.log(.01), requires_grad=True, dtype=torch.float64)
    }

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # learn
    mymod.learn(param_init=param_init, epochs=10)
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

    # Plot embedding if dim==2
    if dim == 2:
        plt.figure(figsize=(7, 7), dpi=600)
        fig, ax = plt.subplots(1, 1)
        ax.set(xlim=[-1, 1])
        ax.set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax.set_title("Final Embedding Sample")
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
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # # make space for internal partials
    # for i in range(S - 1):
    #     partials.append(torch.zeros((1, 4, partials[0].shape[1]), dtype=torch.float64))

    # Compute RAxML tree likelihood
    # TODO: set RAxML to use --JC69. Confirm in log file
    # print('Warning: RAxML using GTR model not JC69')
    # rx = raxml.RaxmlRunner()
    # tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    # peel, blens = utilFunc.dendrophy_to_pb(tree)
    # mats = JC69_p_t(blens)
    # rml_L = calculate_treelikelihood(partials, weights, peel, mats,
    #                                  torch.full([4], 0.25, dtype=torch.float64))
    # print("RAxML Likelihood: " + str(rml_L.item()))
    # print("NB: ELBO is: Likelihood - log(Q) + Jacobian + logPrior(=0)")

    # Get all pair-wise node distance
    # pdm = simtree.phylogenetic_distance_matrix()
    t = Tree(simtree._as_newick_string() + ";")
    nodes = t.get_tree_root().get_descendants(strategy="levelorder")
    dists = [t.get_distance(x, y) for x in nodes for y in nodes]
    dists = np.array(dists).reshape(len(nodes), len(nodes))

    # embed points with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
        emm["r"]), torch.from_numpy(emm["directional"]))
    loc_t0 = p2t0(loc_poin)
    leaf_loc_t0 = loc_t0[:S, :].detach().numpy()
    int_loc_t0 = loc_t0[S:, :].detach().numpy()

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes likewise
    cv = 0.1
    eps = np.finfo(np.double).eps
    leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
    int_sigma = np.log(np.abs(np.array(int_loc_t0)) * cv + eps)
    param_init = {
        "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
        "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
        "int_mu": torch.tensor(int_loc_t0, requires_grad=True, dtype=torch.float64),
        "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64)
    }

    # Plot initial embedding if dim==2
    if dim == 2:
        mymod.learn(param_init=param_init, epochs=0)
        nsamples = 2
        _, ax = plt.subplots(1, 2)
        peels, blens, X = mymod.draw_sample(nsamples)
        ax[0].set(xlim=[-1, 1])
        ax[0].set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('hot')
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax[0], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[0].set_title("Original Embedding Sample")
        plt.close()

    # learn
    mymod.learn(param_init=param_init, epochs=10)

    # # pick a sample and make a tree (Dendropy)
    # dodonaphy_tree_nw = utilFunc.tree_to_newick(
    #     simtree.taxon_namespace.labels(), peels[0], blens[0])
    # dodonaphy_tree_dp = dendropy.Tree.get(
    #     data=dodonaphy_tree_nw, schema="newick")
    # print(dodonaphy_tree_nw)
    # dodonaphy_tree_dp.print_plot()

    # draw the tree samples if dim==2
    if dim == 2:
        peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)
        ax[1].set(xlim=[-1, 1])
        ax[1].set(ylim=[-1, 1])
        for i in range(nsamples):
            utilFunc.plot_tree(
                ax[1], peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax[1].set_title("Final Embedding Sample")
        plt.show()


def test_calculate_likelihood():
    """
    Sometimes calculate_likelihood was throwing errors about
    torch.matmul(Tensor, list), where it wanted torch.matmul(Tensor, Tensor)
    """

    S = 4  # number of sequences to simulate
    seqlen = 100  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Compute RAxML tree
    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    peel, blens = utilFunc.dendrophy_to_pb(tree)
    mats = JC69_p_t(blens)

    # compute partials and weights
    partials, weights = compress_alignment(dna)
    # make space for internal partials
    for i in range(S - 1):
        partials.append(torch.zeros((1, 4, seqlen), dtype=torch.float64))

    _ = calculate_treelikelihood(partials, weights, peel, mats,
                                 torch.full([4], 0.25, dtype=torch.float64))
