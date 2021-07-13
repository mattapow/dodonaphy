import matplotlib.pyplot as plt
import re
from numpy import genfromtxt

experiments = ["simple_geodesics_c5",
               "simple_mst_c5",
               "wrap_mst_c5"]
# experiments = ["simple_mst_c1", "simple_mst_c2", "simple_mst_c5"]
dir = "./data/T6_2"
paths = ("%s/mcmc/%s/mcmc.trees" % (dir, e) for e in experiments)

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

    plt.plot(iter, lnP, linewidth=1)

path = dir + "/beast/mcmc.log"
log = genfromtxt(path, comments="#", skip_header=1)
plt.plot(log[:1000, 0]/10, log[:1000, 2], color='k')
experiments.append("Beast")

plt.xlabel("Iteration")
plt.ylabel("lnP")
plt.legend(experiments)
plt.show()
