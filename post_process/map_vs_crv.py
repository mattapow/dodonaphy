"""
Plot Max A Posterior as a function of curvature.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import MeanShift

name_stub = "simple_nj_c1_d5_crv-"
crv = ("0.10", "0.25", "0.50", "0.75", "1", "1.25", "1.50", "1.75", "2")
burnin = 400
dir = "./data/T17"

fig, ax = plt.subplots(1, 1)
ax.set_yscale('log')

# get MrBayes posterior
fn = "dna.nex.run1.p"
path = os.path.join(dir, "mrbayes", fn)
data = genfromtxt(path, skip_header=2)
posterior = data[burnin:, 1] + data[burnin:, 2]

# get MrBayes MAP
map = MeanShift().fit(posterior.reshape(-1, 1))
mb_base = map.cluster_centers_[0]

paths = [os.path.join(dir, 'mcmc', name_stub + c, 'mcmc.trees') for c in crv]
n_lines = len(paths)

for count, path in enumerate(paths):
    lnL = []
    lnPr = []
    iter = []
    readLine = False
    for i, line in enumerate(open(path)):
        if line.startswith('tree STATE_'):
            _, _, after = line.partition('STATE_')
            iter_, _, _ = after.partition(' ')
            iter.append(int(iter_))

            _, _, after = line.partition('&lnL=')
            lnL_, _, _ = after.partition(', ')
            lnL.append(float(lnL_))

            _, _, after = line.partition('&lnPr=')
            lnPr_, _, _ = after.partition(']')
            lnPr.append(float(lnPr_))

    posterior = np.array(lnPr) + np.array(lnL)
    data = posterior[burnin:].reshape(-1, 1)
    map = MeanShift().fit(data)

    plt.plot(-float(crv[count]), abs(map.cluster_centers_[0]-mb_base)/-mb_base, 'o', color='k')

ax.set_xlabel("Curvature")
ax.set_ylabel("$\mathregular{|MAP_{mb} - MAP_i|/MAP_{mb}}$")
plt.show()
