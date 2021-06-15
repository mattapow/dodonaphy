from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
# from matplotlib.patches import Circle
import numpy as np

"""
Histogram of node locations from MCMC
"""

dir = "./data/T6D2/mcmc"
fp = dir + "/mcmc_locations.csv"

X = genfromtxt(fp)
n = X.shape[1]
n_trees = X.shape[0]
dim = 2  # dimension must be 2 to plot
burnin = 800
if burnin > n_trees:
    raise IndexError("burnin > nuber of trees.")

nodes = [i for i in range(int(n/dim))]
_, ax = plt.subplots(nrows=1, ncols=1)
# circ = Circle((0, 0), radius=1, fill=False, edgecolor='k')
# ax.add_patch(circ)
cmap = matplotlib.cm.get_cmap('Spectral')

for count, node in enumerate(nodes):
    x = X[burnin:, 2*node]
    y = X[burnin:, 2*node+1]
    sns.kdeplot(x=x, y=y, ax=ax, color=cmap(count/len(nodes)))
    ax.annotate('%s' % str(node+1), xy=(float(np.mean(X[burnin:, 2*node])),
                float(np.mean(X[burnin:, 2*node+1]))), xycoords='data')

ax.set_title('Node Embedding Densities from dodonaphy MCMC')
plt.show()
