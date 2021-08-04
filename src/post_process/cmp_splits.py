from numpy import genfromtxt, power, argsort, flip
import matplotlib.pyplot as plt
import os

dir = "./data/T17"
resultsDir = os.path.join(dir, "results")
LODfile = os.path.join(resultsDir, "LOD-table")
BSfile = os.path.join(resultsDir, "partitions.bs")
outfile = os.path.join(resultsDir, "compare-SF")

experiments = ("simple_mst_c5",
               "simple_nj_c5",
               "mrbayes",
               "mrbayes")
n = 4
assert len(experiments) == n
tree1 = os.path.join(dir, 'mcmc', experiments[0], "mcmc.trees")
tree2 = os.path.join(dir, 'mcmc', experiments[1], "mcmc.trees")
treemb1 = os.path.join(dir, experiments[2], "dna.nex.run1.t")
treemb2 = os.path.join(dir, experiments[2], "dna.nex.run2.t")


# call trees-bootstrap from bali-phy
# if not os.path.isdir(resultsDir):
#     os.mkdir(resultsDir)
cmd = "./ext/trees-bootstrap " + tree1 + " " + tree2 + " " + treemb1 + " " + treemb2
cmd = cmd + " --skip=300 --LOD-table=" + LODfile + " > " + BSfile
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
LOD = LOD[:, :N-1]

# sort LOD by first column
ord = flip(argsort(LOD[:, -1]))
LOD = LOD[ord, ]

# create PP table
PP = (power(10, LOD))/(1+power(10, LOD))

# set up the plotting surface
fig, ax = plt.subplots(1, 1)
# cmap = cm.get_cmap('Set1')

# Plot each column as a line
for i in range(n):
    plt.plot(PP[:, i], linewidth=2)

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
plt.legend(experiments)
# plt.xticks([2*i for i in range(10)])

plt.savefig(outfile)
plt.show()
