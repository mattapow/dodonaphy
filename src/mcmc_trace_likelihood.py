import matplotlib.pyplot as plt
import re
from numpy import genfromtxt
# TODO: can we save posterior from MCMC and compare to BEAST's posterior

# experiments = ["mcmc_mst_hot5", "mcmc_tri", "mcmc_mst", "mcmc_geo", "mcmc_mst_sigmoid"]
experiments = ["mcmc_wrap_mst_c5", "mcmc_simple_mst_c5"]
dir = "./data/T6_2"
paths = ("%s/%s/mcmc.trees" % (dir, e) for e in experiments)

for path in paths:
    lnP = []
    iter = []
    pattern = re.compile("&lnP=")
    readLine = False
    for i, line in enumerate(open(path)):
        if line.startswith('tree STATE_'):
            _, _, after = line.partition('STATE_')
            iter_, _, _ = after.partition(' ')
            iter.append(int(iter_))

            _, _, after = line.partition('[&lnP=')
            lnP_, _, _ = after.partition(']')
            lnP.append(float(lnP_))

    plt.plot(iter, lnP, linewidth=.8)

path = dir + "/beast/mcmc.log"
log = genfromtxt(path, comments="#", skip_header=1)

plt.plot(log[:100, 0], log[:100, 2], color='k')
experiments.append("Beast")

plt.xlabel("Iteration")
plt.ylabel("lnP")
plt.legend(experiments)
plt.show()
