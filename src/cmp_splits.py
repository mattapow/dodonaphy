from numpy import genfromtxt, power
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# mthds = ['mcmc', 'logit']
dir = "./data/T6_2"
resultsDir = os.path.join(dir, "results")
LODfile = os.path.join(resultsDir, "LOD-table")
BSfile = os.path.join(resultsDir, "partitions.bs")
outfile = os.path.join(resultsDir, "compare-SF")
experiments = ("beast", "mcmc_wrap_mst_c5", "mcmc_wrap_geodesics_c5", "mcmc_wrap_incentre_c5")
n = 4
assert len(experiments) == n
tree1 = os.path.join(dir, experiments[0], "mcmc.trees")
tree2 = os.path.join(dir, experiments[1], "mcmc.trees")
tree3 = os.path.join(dir, experiments[2], "mcmc.trees")
tree4 = os.path.join(dir, experiments[3], "mcmc.trees")

# call trees-bootstrap from bali-phy
# if not os.path.isdir(resultsDir):
#     os.mkdir(resultsDir)
cmd = "./ext/trees-bootstrap " + tree1 + " " + tree2 + " " + tree3 + " " +\
    tree4 + " --skip=200 --LOD-table=" + LODfile + " > " + BSfile
print(cmd)
print("TODO: redirect output to files")
print("For now, just copy the printed command to the command line, run it, then re-run this file")
# import subprocess
# process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()


# Input LOD file generated from bali-phy's trees-bootstrap


# read file
LOD = genfromtxt(LODfile)
N = LOD.shape[1]
L = LOD.shape[0]

# sort LOD by last column
# ord = flip(argsort(LOD[:, 0]))
# LOD = LOD[ord, ]

# create PP table
PP = (power(10, LOD))/(1+power(10, LOD))

LOD = LOD[:, :N-1]
PP = PP[:, :N-1]

# set up the plotting surface
fig, ax = plt.subplots(1, 1)
cmap = cm.get_cmap('Set1')

# Plot each column as a line
for i in range(n):
    plt.plot(PP[:, i]+i*.01, color=cmap(i/n), linewidth=2)

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
plt.legend(experiments)
# plt.xticks([2*i for i in range(10)])

plt.savefig(outfile)
plt.show()
