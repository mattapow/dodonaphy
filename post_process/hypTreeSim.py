# simulate a hyperbolic tree

import os

import dendropy
import numpy as np
import torch
from dendropy.model.discrete import simulate_discrete_chars
from dodonaphy import base_model, peeler, tree, utils

dim = 2
S = 17
L = 1000
radius = .2
connector = 'geo'

mu = np.zeros((dim))
cov = np.eye(dim)
leaf_locs = np.random.multivariate_normal(mu, cov, S)
leaf_locs = radius * leaf_locs / np.sum(leaf_locs**2, axis=1, keepdims=True)**.5
leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
if connector == 'nj':
    pdm = utils.get_pdm(leaf_r, leaf_dir, astorch=True)
    peel, blens = peeler.nj(pdm)
elif connector == 'hyp_hc':
    peel, int_locs = peeler.make__hard_peel_geodesic(torch.tensor(leaf_locs))
    int_r, int_dir = utils.cart_to_dir(int_locs)
    blens = base_model.BaseModel.compute_branch_lengths(S, peel, leaf_r, leaf_dir, int_r, int_dir, useNP=False)
elif connector == 'geo':
    peel, int_locs = peeler.make_hard_peel_geodesic(torch.tensor(leaf_locs))
    int_r, int_dir = utils.cart_to_dir(int_locs)
    blens = base_model.BaseModel.compute_branch_lengths(S, peel, leaf_r, leaf_dir, int_r, int_dir, useNP=False)

tipnames = np.arange(S).astype(str)
nwk = tree.tree_to_newick(tipnames, peel, blens)

root_dir = "../data/T%s_hypGEO" % (S)
tree_path = os.path.join(root_dir, "simtree.nex")
tree_info_path = os.path.join(root_dir, "simtree.info")
dna_path = os.path.join(root_dir, "dna.nex")
os.mkdir(root_dir)

# convert to dendropy
simtree = dendropy.Tree.get_from_string(nwk, schema='newick')
# simulate dna from tree
dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

# save simtree
simtree.write(path=tree_path, schema="nexus")

# save dna to nexus
dna.write_to_path(dest=dna_path, schema="nexus")

# save simTree info log-likelihood
with open(tree_info_path, 'w', encoding="UTF-8") as f:
    # LL =
    # f.write('Log Likelihood: %f\n' % LL)
    simtree.write_ascii_plot(f)
