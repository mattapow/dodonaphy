import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy import DnaCharacterMatrix
from dodonaphy.model import DodonaphyModel
from dodonaphy.phylo import compress_alignment
from dodonaphy.utils import utilFunc
from matplotlib import pyplot as plt
import matplotlib.cm


def testFunc():
    dim = 3    # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    partials, weights = compress_alignment(dna)

    # print(weights)
    # have to create and convert dist into hyperbolic embedding
    simtree.print_plot()

    mymod = DodonaphyModel(partials, weights, dim)
    mymod.learn(epochs=10)

    nsamples = 3
    peels, blens, X = mymod.draw_sample(nsamples)
    # maximum likelihood parameter values

    # draw the tree samples
    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(1, 1, 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(nsamples):
        utilFunc.plot_tree(
            ax, peels[i], X[i].detach().numpy(), color=cmap(i/nsamples))
    plt.show()


testFunc()
