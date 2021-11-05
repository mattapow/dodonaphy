import matplotlib.cm
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
import numpy as np
import torch
from dodonaphy import peeler, tree, utils

# import pandas as pd

"""
Plot the evolution of node locations from MCMC in the Poincare disk
"""
D = 2  # dimension must be 2 to plot
S = 17
dir = "./data/T" + str(S) + "/mcmc/"
mthd = "simple_mst_c5"
connect_method = 'mst'
fp = dir + mthd + "/locations.csv"

# X = pd.read_csv(fp, header=0, sep=", ")
X = np.genfromtxt(fp, skip_header=1, delimiter=',')
n_trees = X.shape[0]
print('Ensure there values')
n_points = int(X.shape[1]/D)
burnin = 900
sampleEnd = 1000
if sampleEnd > n_trees:
    print(n_trees)
    raise IndexError("requested more than %d number of trees." % n_trees)
print("Double check S=%d and D=%d" % (S, D))
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
    # ax.set_xlim([-.3, .3])
    # ax.set_ylim([-.3, .3])
    leaf_poin = np.zeros((S, D))
    leaf_r = X[j, 0]
    for i in range(S):
        leaf_dir = X[j, 2*i+1:2*i+3]
        leaf_poin[i, :] = (leaf_r * leaf_dir)

    if connect_method in ('incentre', 'geodesics'):
        peel, int_poin = peeler.make_peel_tips(torch.from_numpy(leaf_poin), connect_method)
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
        int_r, int_dir = utils.cart_to_dir(int_poin)
    elif connect_method == 'mst':
        int_poin = np.zeros((S-2, D))
        for i in range(S-2):
            int_r = X[j, i+2*S+1]
            int_dir = X[j, 2*i+3*S-1:2*i+3*S+1]
            int_poin[i, :] = (int_r * int_dir)
            # sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S+i)/n_points))
            # ax.annotate('%s' % str(i+S+1), xy=(int_poin[i, :]), xycoords='data')
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
        int_r, int_dir = utils.cart_to_dir(int_poin)
        peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
    locs = np.concatenate((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, 2)))
    tree.plot_tree(ax, peel, locs, color=(0, 0, 0), labels=True, radius=float(leaf_r[0]))
    plt.title("Tree %d" % j)
    plt.pause(0.05)

plt.show()
