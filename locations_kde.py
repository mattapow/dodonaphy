from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt


"""
Histogram of node locations from MCMC
"""

dir = "./out"
fp = dir + "/mcmc_locations.csv"

X = genfromtxt(fp)
n = X.shape[1]

burnin = 10

nodes = [0, 4, 9]
_, ax = plt.subplots(nrows=len(nodes), ncols=2)

for count, node in enumerate(nodes):
    sns.kdeplot(ax=ax[count, 0], data=X[burnin:, node])
    ax[count, 0].set_ylabel("node " + str(int(node/2+1)) + " / " + str(int(n/2)) + ' x')
    sns.kdeplot(ax=ax[count, 1], data=X[burnin:, node])
    ax[count, 1].set_ylabel("node " + str(int((node-1)/2+1)) + " / " + str(int(n/2)) + ' y')

plt.show()
