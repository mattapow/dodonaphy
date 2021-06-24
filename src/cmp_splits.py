from numpy import genfromtxt, argsort, power, flip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# mthds = ['mcmc', 'logit']
dir = "./data/T6D2_2"
resultsDir = os.path.join(dir, "results")
LODfile = os.path.join(resultsDir, "LOD-table")
BSfile = os.path.join(resultsDir, "partitions.bs")
outfile = os.path.join(resultsDir, "compare-SF")
tree0 = os.path.join(dir, "beast/dna.trees")
tree1 = os.path.join(dir, "mcmc_step05_biginit/mcmc.trees")
tree2 = os.path.join(dir, "logit_lr0001/vi.trees")
tree3 = os.path.join(dir, "mcmc_geo/mcmc.trees")
tree4 = os.path.join(dir, "logit_geo/vi.trees")
n = 5

# call trees-bootstrap from bali-phy
makeBoots = True
if makeBoots:
    # if not os.path.isdir(resultsDir):
    #     os.mkdir(resultsDir)
    import subprocess
    cmd = "./ext/trees-bootstrap " + tree0 + " " + tree1 + " " + tree2 + " " +\
        tree3 + " " + tree4 + " --LOD-table=" + LODfile + " > " + BSfile
    print(cmd)
    # TODO: redirect output to files
    # For now, just copy the printed command to the command line, run it, then re-run this file
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


# Input LOD file generated from bali-phy's trees-bootstrap


# read file
LOD = genfromtxt(LODfile)
N = LOD.shape[1]
L = LOD.shape[0]

# sort LOD by last column
ord = flip(argsort(LOD[:, 0]))
LOD = LOD[ord, ]

# create PP table
PP = (power(10, LOD))/(1+power(10, LOD))

LOD = LOD[:, :N-1]
PP = PP[:, :N-1]

# set up the plotting surface
fig, ax = plt.subplots(1, 1)
cmap = cm.get_cmap('Set1')

# Plot each column as a line
for i in range(n):
    plt.plot(PP[:, i], color=cmap(i/n), linewidth=2)

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
# plt.legend(('BEAST', '%s' % mthd))
plt.legend(('BEAST', 'mcmc-mst', 'vi logit-mst', 'mcmc-geo', 'vi logit-geo'))
# plt.xticks([2*i for i in range(10)])

plt.savefig(outfile)
plt.show()
