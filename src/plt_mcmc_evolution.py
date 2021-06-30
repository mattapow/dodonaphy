from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm
# from matplotlib.patches import Circle
import numpy as np
from dodonaphy.src import peeler, tree
import torch

"""
Plot the evolution of node locations from MCMC in the Poincare disk
"""

dir = "./data/T6D2_2/"
mthd = "mcmc_tri"
fp = dir + mthd + "/locations.csv"

X = genfromtxt(fp)
n_trees = X.shape[0]
D = 2  # dimension must be 2 to plot
S = 6
n_points = int(X.shape[1]/D)
burnin = 900
sampleEnd = 1000
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
    ax.set_xlim([-.2, .2])
    ax.set_ylim([-.2, .2])
    leaf_poin = np.zeros((S, D))
    for i in range(S):
        x = X[j, 2*i]
        y = X[j, 2*i+1]
        x = 1/(1+np.exp(-x)) * 2 - 1
        y = 1/(1+np.exp(-y)) * 2 - 1
        leaf_poin[i, :] = (x, y)
    peel, int_poin = peeler.make_peel_incentre(torch.from_numpy(leaf_poin))
    locs = np.concatenate((leaf_poin, int_poin))
    tree.plot_tree(ax, peel, locs, color=(0, 0, 0), labels=True)
    plt.pause(0.05)

plt.show()
