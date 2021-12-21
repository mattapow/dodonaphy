"""
Node locations from VI
"""
import os

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dodonaphy import Chyperboloid, peeler, tree, utils
from dodonaphy.vi import read
from numpy.random import multivariate_normal

path_read = "./data/T17_long_tree/vi_noGrad/simple_nj_lr2_k2_d5"
fn = os.path.join(path_read, "VI_params_init.csv")
connector = "geodesics"
embedder = "wrap"
D = 2
var_params = read(fn, internals=False)
S = int(len(var_params["leaf_mu"][0]))
print("Ensure these values: D=%d, S=%d" % (D, S))

N = 2 * S - 2
if connector == "geodesics":
    N = S

nodes = [i for i in range(N)]
_, ax = plt.subplots(nrows=1, ncols=1)

cmap = matplotlib.cm.get_cmap("Spectral")

leaf_poin = np.zeros((S, D))
int_poin = np.zeros((S - 2, D))
n_samples = 100

# plot tips
for node in range(S):
    loc = var_params["leaf_mu"][:, node]
    cov = np.exp(var_params["leaf_sigma"][:, node]) * np.eye(D)

    # sample from normal and convert to poincare ball
    data = multivariate_normal(loc, cov, n_samples)
    if embedder == "simple":
        data_poin = utils.real2ball(torch.from_numpy(data), D)
        x = data_poin[:, 0]
        y = data_poin[:, 1]
        mu = (torch.mean(x), torch.mean(y))
    elif embedder == "wrap":
        x = torch.tensor(data[:, 0])
        y = torch.tensor(data[:, 1])
        mu = (torch.mean(x), torch.mean(y))
    # if node == 1 or node == 4:
    #    mu += np.array([.1, .1])

    sns.kdeplot(x=x, y=y, ax=ax, color=cmap(node / N), thresh=0.1)
    # ax.annotate('%s' % str(node+1), xy=(float(np.mean(x)), float(np.mean(y))), xycoords='data')
    leaf_poin[node, :] = mu

# plot internal nodes
if connector == "mst":
    for node in range(S - 2):
        loc = var_params["int_mu"][node * D : D * (node + 1)]
        cov = np.exp(var_params["int_sigma"][node * D : D * (node + 1)]) * np.eye(D)
        data = multivariate_normal(loc, cov, n_samples)

        if embedder == "simple":
            data_poin = utils.real2ball(torch.from_numpy(data), D)
            x = data_poin[:, 0]
            y = data_poin[:, 1]
            mu = utils.real2ball(torch.from_numpy(loc), D)
        elif embedder == "wrap":
            x = Chyperboloid.t02p(torch.tensor(data[:, 0]), D)
            y = Chyperboloid.t02p(torch.tensor(data[:, 1]), D)

        sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S + node) / N))
        # ax.annotate('%s' % str(S+node+1), xy=(float(np.mean(x)), float(np.mean(y))), xycoords='data')
        int_poin[node, :] = (x[-1], y[-1])

    # make peel
    leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
    int_r, int_dir = utils.cart_to_dir(int_poin)
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

elif connector == "geodesics":
    # make peel
    peel, int_poin = peeler.make_peel_tips(torch.tensor(leaf_poin), connector=connector)
    leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
    int_r, int_dir = utils.cart_to_dir(int_poin)

# plot tree of mean points
X = np.concatenate((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, D)))
tree.plot_tree(ax, peel, X, color=(0, 0, 0), labels=True)

# ax.set_xlim([-.3, .35])
# ax.set_ylim([-.35, .3])


ax.set_title("%s, %s Node Embedding Densities." % (connector, embedder))
plt.show()
