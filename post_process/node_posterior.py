"""
Each draw from the sample should be different in likelihood.
"""
# from torch.functional import meshgrid
import dendropy

# from dendropy.simulate import treesim
# from dendropy.model.discrete import simulate_discrete_chars
import matplotlib.pyplot as plt
import numpy as np
import torch
from dodonaphy import Chyperboloid, peeler, tree, utils
from dodonaphy.phylo import compress_alignment
from dodonaphy.vi import DodonaphyVI

# from matplotlib import cm
# import seaborn as sns


dim = 2  # number of dimensions for embedding
S = 6  # number of sequences to simulate
seqlen = 1000  # length of sequences to simulate
prior = {"birth_rate": 2.0, "death_rate": 0.5}

# read simulated a tree
path_write = "./data/T%d_2" % (S)
treePath = "%s/simtree.nex" % path_write
dnaPath = "%s/dna.nex" % path_write
simtree = dendropy.Tree.get(path=treePath, schema="nexus")
dna = dendropy.DnaCharacterMatrix.get(path=dnaPath, schema="nexus")

# Initialise model
partials, weights = compress_alignment(dna)
mymod = DodonaphyVI(partials, weights, dim, embedder="wrap", connector="geodesics")

# specify locations
leaf_t0 = torch.tensor(
    [
        -0.1804018,
        0.06084276,
        -0.03132725,
        -0.01874877,
        -0.10155862,
        0.12463426,
        -0.08097307,
        0.09621198,
        -0.0098637,
        0.04274979,
        -0.1147425,
        0.12270321,
    ],
    dtype=torch.float64,
).reshape((S, dim))
int_t0 = torch.tensor(
    [
        -0.00433582,
        0.45464002,
        0.04688382,
        0.02818116,
        -0.03721131,
        0.01916582,
        -0.08877883,
        0.07997011,
    ],
    dtype=torch.float64,
).reshape((S - 2, dim))

# convert to poincare ball
leaf_poin = Chyperboloid.t02p(leaf_t0, dim).reshape((S, dim))
int_poin = Chyperboloid.t02p(int_t0, dim).reshape((S - 2, dim))

# move one node and compute posterior
steps = 100
X = np.linspace(-0.4, 0.4, steps)
Y = np.linspace(-0.4, 0.4, steps)
lnPost = np.zeros((steps, steps))
idx = 0
best_lnPost = -np.inf
best_loc = np.zeros((1, 2))
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        int_poin[idx, :] = torch.tensor([x, y])
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
        int_r, int_dir = utils.cart_to_dir(int_poin)

        peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
        blen = mymod.compute_branch_lengths(S, peel, leaf_r, leaf_dir, int_r, int_dir)
        lnLike = mymod.compute_LL(peel, blen)
        lnPrior = mymod.compute_prior_gamma_dir(blen)
        lnPost_ = lnLike + lnPrior
        lnPost[j, i] = lnPost_
        if lnPost_ > best_lnPost:
            best_lnPost = lnPost_
            best_loc = torch.tensor([x, y])


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(X, X)
# ax.plot_surface(X, Y, lnPost, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# hardcode and plot best embedding
fig, ax = plt.subplots(1, 1)
int_poin[idx, :] = best_loc
print(best_lnPost)
print(best_loc)
leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
int_r, int_dir = utils.cart_to_dir(int_poin)
peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
locs = torch.cat((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, dim)))
tree.plot_tree(ax, peel, locs)

# contour plot of best positions
X, Y = np.meshgrid(X, Y)
plt.contourf(X, Y, lnPost, cmap="hot", levels=200)
plt.colorbar()
plt.show()
