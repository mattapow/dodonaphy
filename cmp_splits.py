from numpy import genfromtxt, argsort, power, flip
import matplotlib.pyplot as plt

# Input LOD file generated from bali-phy's trees-bootstrap
mthd = 'mcmc'
filename = 'data/Taxa6Dim2Boosts1/' + mthd + '/results/LOD-table'
outfile = 'compare-SF.png'

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

# Plot first column as a red line
plt.plot(PP[:, 0], color=[1, 0, 0], linewidth=2)

# Plot second column as a blue line
plt.plot(PP[:, 1], color=[0, 0, 1], linewidth=2)

plt.ylabel('Split Posterior Probability')
plt.xlabel('Split index')
plt.legend(('BEAST', '%s' % mthd))

plt.show()
