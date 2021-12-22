import os

import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import arange, argsort, flip, genfromtxt, power

ds_index = 1
dir = "../analysis/DS" + str(ds_index)

experiments = ["simple_nj/d5_c4_free_r_million"]
n = len(experiments)

cmd = "../ext/trees-bootstrap "
for i in range(n):
    tree_str = os.path.join(dir, 'mcmc', experiments[i], "mcmc.trees")
    path_split = os.path.join(dir, "mcmc", experiments[i], "results", "splits")
    os.mkdir(path_split)
    LODfile = os.path.join(path_split, "LOD-table")
    BSfile = os.path.join(path_split, "partitions.bs")
    outfile = os.path.join(path_split, "compare-SF")
    cmd += tree_str + " "
cmd += os.path.join(dir, "mb", f"DS{str(ds_index)}.nex.run1.t") + " "
cmd += os.path.join(dir, "mb", f"DS{str(ds_index)}.nex.run2.t")
cmd += " --skip=300 --LOD-table=" + LODfile + " > " + BSfile

# call trees-bootstrap from bali-phy
print("Run this command:")
print("")
print(cmd)
print("\nFor now, just copy the printed command to the command line, run it, then re-run this file")
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
cmap = cm.get_cmap('viridis', n)
handles = []

# Plot each column as a line
idx = arange(L)
for i in range(n):
    plt.scatter(idx+i/L, PP[:, i], zorder=n-i+2, color=cmap.colors[i, :], label=experiments[i])
    for j in range(L):
        plt.plot([idx[j], idx[j]], [PP[j, i], PP[j, -2]], zorder=n-i+2, color=cmap.colors[i, :])
plt.plot(PP[:, -2], linewidth=2, zorder=0, color='k', label="MrBayes run 1")
plt.plot(PP[:, -1], linewidth=2, zorder=0, color='k', label="MrBayes run 2")

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
plt.legend()

plt.savefig(outfile)
plt.show()
