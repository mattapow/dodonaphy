from src.model import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from src.hyperboloid import p2t0

import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from ete3 import Tree

"""
Initialise int and lead embeddings with hydra
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
