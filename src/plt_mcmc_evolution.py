from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm
# from matplotlib.patches import Circle
import numpy as np
from dodonaphy.src import peeler, tree, utils
import torch

"""
Plot the evolution of node locations from MCMC in the Poincare disk
"""

dir = "./data/T6_2/"
mthd = "mcmc_mst_scale000001_1"
fp = dir + mthd + "/locations.csv"
incentre = False

X = genfromtxt(fp)
n_trees = X.shape[0]
D = 2  # dimension must be 2 to plot
S = 6
n_points = int(X.shape[1]/D)
burnin = 0
sampleEnd = 30
if sampleEnd > n_trees:
    print(n_trees)
    raise IndexError("requested more than nuber of trees.")
# if not mthd == "mcmc":
#     burnin = 0
#     sampleEnd = 1000

_, ax = plt.subplots(nrows=1, ncols=1)
# circ = Circle((0, 0), radius=1, fill=False, edgecolor='k')
# ax.add_patch(circ)
cmap = matplotlib.cm.get_cmap('Spectral')
ax.set_title('%s Node Embedding Densities. Trees %d to %d' % (mthd, burnin, sampleEnd))

for j in range(burnin, sampleEnd):
    plt.cla()
    ax.set_xlim([-.3, .3])
    ax.set_ylim([-.3, .3])
    leaf_poin = np.zeros((S, D))
    for i in range(S):
        x = X[j, 2*i]
        y = X[j, 2*i+1]
        x = 1/(1+np.exp(-x)) * 2 - 1
        y = 1/(1+np.exp(-y)) * 2 - 1
        leaf_poin[i, :] = (x, y)

    if incentre:
        peel, int_poin = peeler.make_peel_incentre(torch.from_numpy(leaf_poin))
    else:
        int_poin = np.zeros((S-2, D))
        for i in range(S-2):
            x = X[burnin:sampleEnd, 2*i+2*S]
            y = X[burnin:sampleEnd, 2*i+2*S+1]
            x = 1/(1+np.exp(-x)) * 2 - 1
            y = 1/(1+np.exp(-y)) * 2 - 1
            int_poin[i, :] = (np.mean(x), np.mean(y))
            # sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S+i)/n_points))
            # ax.annotate('%s' % str(i+S+1), xy=(int_poin[i, :]), xycoords='data')
        leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(np.concatenate((leaf_poin, int_poin)))
        peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    locs = np.concatenate((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, 2)))
    tree.plot_tree(ax, peel, locs, color=(0, 0, 0), labels=True)
    plt.pause(0.05)

plt.show()
