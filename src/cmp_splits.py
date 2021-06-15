from numpy import genfromtxt, argsort, power, flip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Input LOD file generated from bali-phy's trees-bootstrap
mthd = 'mcmc'
filename = 'data/T6D2/results/LOD-table'
outfile = 'data/T6D2/results/compare-SF'

# read file
LOD = genfromtxt(filename)
N = LOD.shape[1]
L = LOD.shape[0]

# sort LOD by last column
ord = flip(argsort(LOD[:, 0]))
LOD = LOD[ord, ]

# create PP table
PP = (power(10, LOD))/(1+power(10, LOD))

# LOD = LOD[:, 1:N-1]
# PP = PP[:, 1:N-1]

# set up the plotting surface
fig, ax = plt.subplots(1, 1)
cmap = cm.get_cmap('Set1')

# Plot each column as a line
for i in range(5):
    plt.plot(PP[:, i], color=cmap(i/5), linewidth=2)

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
# plt.legend(('BEAST', '%s' % mthd))
plt.legend(('BEAST', 'mcmc', 'vi_logit', 'vi_wrap', 'vi_wrap_B3'))
plt.xticks([2*i for i in range(10)])

plt.savefig(outfile)
plt.show()
