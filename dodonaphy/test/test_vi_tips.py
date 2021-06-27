import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.vi_tips import VITips as vi
from src.phylo import compress_alignment
import torch


def test_draws_different():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = vi(partials, weights, dim)

    # learn
    mymod.learn(epochs=1, path_write=None)

    # draw
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)
    assert not torch.equal(blens[0], blens[1])
    assert not torch.equal(blens[0], blens[2])
    assert not torch.equal(blens[1], blens[2])

    assert not torch.equal(lp__[0], lp__[1])
    assert not torch.equal(lp__[0], lp__[2])
    assert not torch.equal(lp__[1], lp__[2])
