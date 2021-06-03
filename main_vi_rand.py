import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.model import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from matplotlib import pyplot as plt
import matplotlib.cm

""" Dodonaphy model with randomly initialized parameters for variational inference
"""
dim = 2    # number of dimensions for embedding
nseqs = 6  # number of sequences to simulate
seqlen = 1000  # length of sequences to simulate

# simulate a tree
simtree = treesim.birth_death_tree(birth_rate=2.0, death_rate=0.5, num_extant_tips=nseqs)
dna = simulate_discrete_chars(seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

# model initiation and training
partials, weights = compress_alignment(dna)
mymod = DodonaphyModel(partials, weights, dim)

# variational parameters: [default] randomly generated within model constructor
mymod.learn(epochs=10)

if dim == 2:
    # draw samples from variational posterior
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

    # plot sample trees
    _, ax = plt.subplots(1, 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(nsamples):
        utilFunc.plot_tree(ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
    plt.show()
