import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.model import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from src.hyperboloid import p2t0
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from dendropy.interop import raxml
import random

"""
Initialise the emebedding with RAxML distances given to hydra.
RAxML gives internal nodes as well.
"""

dim = 2  # number of dimensions for embedding
nseqs = 6  # number of sequences to simulate
seqlen = 1000  # length of sequences to simulate

# # simulate a tree
rng = random.Random(1)
simtree = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs, rng=rng)
dna = simulate_discrete_chars(seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

# s = simtree.postorder_node_iter()
# testing raxml
# rx = raxml.RaxmlRunner(raxml_path="raxmlHPC-AVX2")
# # tree = rx.estimate_tree(char_matrix=dna, raxml_args=['-e', 'likelihoodEpsilon', '-h' '--JC69'])
# tree = rx.estimate_tree(char_matrix=dna, raxml_args=["-h", "--JC69"])

rx = raxml.RaxmlRunner()
rxml_tree = rx.estimate_tree(char_matrix=dna)
assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
dist = np.array([[assemblage_data[i][j] for j in sorted(
    assemblage_data[i])] for i in sorted(assemblage_data)])
emm = utilFunc.hydra(D=dist, dim=dim, equi_adj=0., stress=True)
print('stress = ' + str(emm["stress"]))

leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
    emm["r"]), torch.from_numpy(emm["directional"]))
leaf_loc_t0 = p2t0(leaf_loc_poin).detach().numpy()

# set initial leaf positions from hydra with small coefficient of variation
# set internal nodes to narrow distributions at origin
cv = 1. / 50
eps = np.finfo(np.double).eps
leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
param_init = {
    "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
    "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
    "int_mu": torch.zeros(nseqs - 2, dim, requires_grad=True, dtype=torch.float64),
    "int_sigma": torch.full((nseqs - 2, dim), np.log(.01), requires_grad=True, dtype=torch.float64)
}

# Initialise model
partials, weights = compress_alignment(dna)
mymod = DodonaphyModel(partials, weights, dim)

# learn
mymod.learn(param_init=param_init, epochs=10)
nsamples = 3
peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

# Plot embedding if dim==2
if dim == 2:
    _, ax = plt.subplots(1, 1)
    ax.set(xlim=[-1, 1])
    ax.set(ylim=[-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(nsamples):
        utilFunc.plot_tree(
            ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
    ax.set_title("Final Embedding Sample")
    plt.show()
