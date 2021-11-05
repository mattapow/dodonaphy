import os

import dendropy
import torch
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from numpy import allclose
from dodonaphy import vi
from dodonaphy.phylo import compress_alignment
from dodonaphy.vi import DodonaphyVI


def test_draws_different_vi_simple_incentre():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs
    )
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, dim, embed_method="simple", connect_method="incentre"
    )

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


def test_draws_different_vi_simple_geodesics():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs
    )
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, dim, embed_method="simple", connect_method="geodesics"
    )

    # learn
    # torch.autograd.set_detect_anomaly(True)
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


def test_draws_different_vi_simple_nj():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs
    )
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, dim, embed_method="simple", connect_method="nj"
    )

    # learns
    torch.autograd.set_detect_anomaly(True)
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


def test_io():
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=6
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, 2, embed_method="simple", connect_method="nj"
    )
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    fp = os.path.join(tmp_dir, "test_data.csv")
    if os.path.exists(fp):
        os.remove(fp)
    mymod.save(fp)
    output = vi.read(fp, connect_method="nj")
    assert allclose(
        output["leaf_mu"], mymod.VariationalParams["leaf_mu"].detach().numpy()
    )
    assert allclose(
        output["leaf_sigma"], mymod.VariationalParams["leaf_sigma"].detach().numpy()
    )
    os.remove(fp)
    os.removedirs(tmp_dir)
