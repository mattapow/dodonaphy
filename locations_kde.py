from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt


"""
Histogram of node locations from MCMC
"""

dir = "./out"
fn = dir + "/locations.txt"

X = genfromtxt(fn)
n = X.shape[1]

for i in range(n):
    sns.kdeplot(X[:, i])
    if i % 2 == 0:
        plt.title("node " + str(int(i/2+1)) + " / " + str(int(n/2)) + ' x')
    else:
        plt.title("node " + str(int((i-1)/2+1)) + " / " + str(int(n/2)) + ' y')
    plt.show()
