# simulate a hyperbolic tree

import dendropy
import numpy as np
from dodonaphy.src import peeler, tree, utils
import os
from dendropy.model.discrete import simulate_discrete_chars

dim = 3
S = 17
L = 1000
radius = .3

mu = np.zeros((dim))
cov = np.eye(dim)
leaf_locs = np.random.multivariate_normal(mu, cov, S)
leaf_locs = radius * leaf_locs / np.sum(leaf_locs**2, axis=1, keepdims=True)**.5
leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
pdm = utils.get_pdm(leaf_r, leaf_dir, astorch=True)
peel, blens = peeler.nj(pdm)

tipnames = np.arange(S).astype(str)
nwk = tree.tree_to_newick(tipnames, peel, blens)

root_dir = "./data/T%s_hypNJ" % (S)
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
with open(tree_info_path, 'w') as f:
    # LL =
    # f.write('Log Likelihood: %f\n' % LL)
    simtree.write_ascii_plot(f)
