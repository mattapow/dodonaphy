from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
# from matplotlib.patches import Circle
import numpy as np

"""
Histogram of node locations from MCMC
"""

dir = "../data/T6D2/"
mthd = "mcmc"
fp = dir + mthd + "/locations.csv"

X = genfromtxt(fp)
n = X.shape[1]
n_trees = X.shape[0]
dim = 2  # dimension must be 2 to plot
burnin = 500
sampleEnd = 600
if sampleEnd > n_trees:
    raise IndexError("burnin > nuber of trees.")
if not mthd == "mcmc":
    burnin = 0
    sampleEnd = -1

nodes = [i for i in range(int(n/dim))]
_, ax = plt.subplots(nrows=1, ncols=1)
# circ = Circle((0, 0), radius=1, fill=False, edgecolor='k')
# ax.add_patch(circ)
cmap = matplotlib.cm.get_cmap('Spectral')

for count, node in enumerate(nodes):
    x = X[burnin:sampleEnd, 2*node]
    y = X[burnin:sampleEnd, 2*node+1]
    sns.kdeplot(x=x, y=y, ax=ax, color=cmap(count/len(nodes)))
    ax.annotate('%s' % str(node+1), xy=(float(np.mean(X[burnin:sampleEnd, 2*node])),
                float(np.mean(X[burnin:sampleEnd, 2*node+1]))), xycoords='data')

ax.set_xlim([-1, .75])
ax.set_ylim([-1, 1.05])


ax.set_title('%s Node Embedding Densities. Trees %d-%d' % (mthd, burnin, sampleEnd))
plt.show()
