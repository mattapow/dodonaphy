import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
import os
import numpy as np


experiments_mcmc = ["simple_mst_c5",
                    "simple_nj_c5"]
burnin = 300
experiments_vi = []
dir = "./data/T6"
paths = [os.path.join(dir, 'mcmc', e, 'mcmc.trees') for e in experiments_mcmc]
for e in experiments_vi:
    paths.append("%s/vi/%s/vi.trees" % (dir, e))

fig, ax = plt.subplots(3, 2)

# path = os.path.join(dir, "beast/mcmc.log")
# data = genfromtxt(path, comments="#", skip_header=1)
# ax[0].plot(data[:1000, 0]/10, data[:1000, 1])
# sns.kdeplot(data[burnin:1000, 1], ax=ax[1])

path = os.path.join(dir, "mrbayes/dna.nex.run1.p")
data = genfromtxt(path, skip_header=2)
posterior = data[:, 1]
ax[0, 0].plot(data[:1000, 0]/50, posterior[:1000])
sns.kdeplot(posterior[burnin:], ax=ax[0, 1])
posterior = data[:, 2]
ax[1, 0].plot(data[:1000, 0]/50, posterior[:1000])
sns.kdeplot(posterior[burnin:], ax=ax[1, 1])
posterior = data[:, 1] + data[:, 2]
ax[2, 0].plot(data[:1000, 0]/50, posterior[:1000])
sns.kdeplot(posterior[burnin:], ax=ax[2, 1])

experiments_mcmc.insert(0, "MrBayes")
# experiments_mcmc.insert(0, 'Beast')

for path in paths:
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

    posterior = np.array(lnL)
    ax[0, 0].plot(iter, posterior, linewidth=1)
    sns.kdeplot(posterior[burnin:], ax=ax[0, 1])
    posterior = np.array(lnPr)
    ax[1, 0].plot(iter, posterior, linewidth=1)
    sns.kdeplot(posterior[burnin:], ax=ax[1, 1])
    posterior = np.array(lnPr) + np.array(lnL)
    ax[2, 0].plot(iter, posterior, linewidth=1)
    sns.kdeplot(posterior[burnin:], ax=ax[2, 1])

for i in range(3):
    ax[i, 0].set_xlabel("Iteration")
    ax[i, 1].set_ylabel('Density')

ax[0, 0].set_ylabel("Log Likelihood")
ax[0, 1].set_xlabel('Log Likelihood')
ax[1, 0].set_ylabel("Log Prior")
ax[1, 1].set_xlabel('Log Prior')
ax[2, 0].set_ylabel("Log Posterior")
ax[2, 1].set_xlabel('Log Posterior')


plt.legend(experiments_mcmc)
plt.show()
