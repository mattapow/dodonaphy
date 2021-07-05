import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from dodonaphy.src.vi import read
from dodonaphy.src import utils, peeler, tree
from numpy.random import multivariate_normal
import os
import torch

"""
Node locations from VI
"""

path_read = "./data/T6_2/logit_mst_lr4_k10"
fn = os.path.join(path_read, "VI_params_init.csv")
connect_method = 'mst'
embed_method = 'logit'
var_params = read(fn, connect_method=connect_method)
D = 2

S = int(len(var_params['leaf_mu'])/D)
N = 2*S - 2
if connect_method == 'geodesics':
    N = S
print('Ensure these values: D=%d, S=%d' % (D, S))

nodes = [i for i in range(N)]
_, ax = plt.subplots(nrows=1, ncols=1)

cmap = matplotlib.cm.get_cmap('Spectral')

leaf_poin = np.zeros((S, D))
int_poin = np.zeros((S-2, D))
n_samples = 1000

# plot tips
for node in range(S):
    loc = var_params['leaf_mu'][node*D:D*(node+1)]
    cov = np.exp(var_params['leaf_sigma'][node*D:D*(node+1)])*np.eye(D)

    # sample from normal and convert using sigmoid function
    data = multivariate_normal(loc, cov, n_samples)
    if embed_method == 'logit':
        x = 1/(1+np.exp(-data[:, 0])) * 2 - 1
        y = 1/(1+np.exp(-data[:, 1])) * 2 - 1
        mu = 1/(1+np.exp(-loc)) * 2 - 1
    elif embed_method == 'wrap':
        x = data[:, 0]
        y = data[:, 1]
        mu = (np.mean(x), np.mean(y))
    # if node == 1 or node == 4:
    #    mu += np.array([.1, .1])

    sns.kdeplot(x=x, y=y, ax=ax, color=cmap(node/N), thresh=.1)
    # ax.annotate('%s' % str(node+1), xy=(float(np.mean(x)), float(np.mean(y))), xycoords='data')
    leaf_poin[node, :] = mu

# plot internal nodes
if connect_method == 'mst':
    for node in range(S-2):
        loc = var_params['int_mu'][node*D:D*(node+1)]
        cov = np.exp(var_params['int_sigma'][node*D:D*(node+1)])*np.eye(D)

        # sample from normal and convert using sigmoid function
        data = multivariate_normal(loc, cov, n_samples)
        x = 1/(1+np.exp(-data[:, 0])) * 2 - 1
        y = 1/(1+np.exp(-data[:, 1])) * 2 - 1
        # mu = 1/(1+np.exp(-loc)) * 2 - 1

        sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S+node)/N))
        # ax.annotate('%s' % str(S+node+1), xy=(float(np.mean(x)), float(np.mean(y))), xycoords='data')
        int_poin[node, :] = (x[-1], y[-1])

    # make peel
    leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
    int_r, int_dir = utils.cart_to_dir(int_poin)
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

elif connect_method == 'geodesics':
    # make peel
    peel, int_poin = peeler.make_peel_incentre(torch.tensor(leaf_poin))
    leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
    int_r, int_dir = utils.cart_to_dir(int_poin)

# plot tree of mean points
X = np.concatenate((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, D)))
tree.plot_tree(ax, peel, X, color=(0, 0, 0), labels=True)

# ax.set_xlim([-.3, .35])
# ax.set_ylim([-.35, .3])


ax.set_title('%s, %s Node Embedding Densities.' % (connect_method, embed_method))
plt.show()
print('done')
