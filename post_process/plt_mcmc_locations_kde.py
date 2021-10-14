import matplotlib.cm
# import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
import numpy as np
import torch
from src import peeler, tree, utils
from numpy import genfromtxt

"""
Histogram of node locations from MCMC
"""

dir = "./data/T6_2/"
mthd = "mcmc/simple_mst_c1"
fp = dir + mthd + "/locations.csv"
isGeodesics = False

X = genfromtxt(fp, dtype=np.double)
n_trees = X.shape[0]
D = 2  # dimension must be 2 to plot
S = 6
n_points = int(X.shape[1]/D)
burnin = 980
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

leaf_poin = np.zeros((S, D))
for i in range(S):
    x = X[burnin:sampleEnd, 2*i]
    y = X[burnin:sampleEnd, 2*i+1]
    for j, xj in enumerate(x):
        x[j, :] = utils.real2ball(torch.from_numpy(xj), D)
    for j, yj in enumerate(y):
        y[j, :] = utils.real2ball(torch.from_numpy(yj), D)
    idx = -1
    leaf_poin[i, :] = (x[idx], y[idx])
    # sns.kdeplot(x=x, y=y, ax=ax, color=cmap(i/n_points), thresh=.1)
    # ax.annotate('%s' % str(i+1), xy=(leaf_poin[i, :]), xycoords='data')


# ax.set_xlim([-1, .75])
# ax.set_ylim([-1, 1.05])

# plot tree of mean points
if isGeodesics:
    peel, int_poin = peeler.make_peel_incentre(torch.from_numpy(leaf_poin))
    X = np.concatenate((leaf_poin, int_poin))
else:
    int_poin = np.zeros((S-2, D))
    for i in range(S-2):
        x = X[burnin:sampleEnd, 2*i+2*S]
        y = X[burnin:sampleEnd, 2*i+2*S+1]
        for j, xj in enumerate(x):
            x[j, :] = utils.real2ball(torch.from_numpy(xj), D)
        for j, yj in enumerate(y):
            y[j, :] = utils.real2ball(torch.from_numpy(yj), D)
        int_poin[i, :] = (np.mean(x), np.mean(y))
        # sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S+i)/n_points))
        # ax.annotate('%s' % str(i+S+1), xy=(int_poin[i, :]), xycoords='data')
    X = np.concatenate((leaf_poin, int_poin, leaf_poin[0].reshape(1, D)))
    leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(np.concatenate((leaf_poin, int_poin)))
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
tree.plot_tree(ax, peel, X, color=(0, 0, 0), labels=True)

ax.set_title('%s Node Embedding Densities. Trees %d to %d' % (mthd, burnin, sampleEnd))
plt.show()
