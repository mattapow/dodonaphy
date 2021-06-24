from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
# from matplotlib.patches import Circle
import numpy as np
from dodonaphy.src.utils import utilFunc

"""
Histogram of node locations from MCMC
"""

dir = "./data/T6D2_2/"
mthd = "mcmc_geo"
fp = dir + mthd + "/locations.csv"

X = genfromtxt(fp)
n = X.shape[1]
n_trees = X.shape[0]
D = 2  # dimension must be 2 to plot
S = int((n/D+2)/2)
burnin = 500
sampleEnd = 600
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
    x = 1/(1+np.exp(-x)) * 2 - 1
    y = 1/(1+np.exp(-y)) * 2 - 1
    leaf_poin[i, :] = (np.mean(x), np.mean(y))
    sns.kdeplot(x=x, y=y, ax=ax, color=cmap(i/n))
    ax.annotate('%s' % str(i+1), xy=(leaf_poin[i, :]), xycoords='data')

int_poin = np.zeros((S-2, D))
for i in range(S-2):
    x = X[burnin:sampleEnd, 2*i+2*S]
    y = X[burnin:sampleEnd, 2*i+2*S+1]
    x = 1/(1+np.exp(-x)) * 2 - 1
    y = 1/(1+np.exp(-y)) * 2 - 1
    int_poin[i, :] = (np.mean(x), np.mean(y))
    sns.kdeplot(x=x, y=y, ax=ax, color=cmap((S+i)/n))
    ax.annotate('%s' % str(i+S+1), xy=(int_poin[i, :]), xycoords='data')

# ax.set_xlim([-1, .75])
# ax.set_ylim([-1, 1.05])

leaf_r, leaf_dir = utilFunc.cart_to_dir(leaf_poin)
int_r, int_dir = utilFunc.cart_to_dir(int_poin)
peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)

# plot tree of mean points
X = np.concatenate((leaf_poin, int_poin, leaf_poin[0].reshape(1, D)))
utilFunc.plot_tree(ax, peel, X, color=(0, 0, 0), labels=False)

ax.set_title('%s Node Embedding Densities. Trees %d to %d' % (mthd, burnin, sampleEnd))
plt.show()
