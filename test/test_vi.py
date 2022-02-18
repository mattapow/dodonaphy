import os

import dendropy
import torch
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from numpy import allclose
from dodonaphy import vi
from dodonaphy.phylo import compress_alignment
from dodonaphy.vi import DodonaphyVI


def test_draws_different_vi_project_up_geodesics():
    """Each draw from the sample should be different in likelihood."""
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=6
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, dim=2, embedder="up", connector="geodesics", soft_temp=1e-8
    )

    mymod.learn(epochs=2, path_write=None)

    _, blens, _, lp__ = mymod.draw_sample(3, lp=True)
    assert not torch.equal(blens[0], blens[1])
    assert not torch.equal(blens[0], blens[2])
    assert not torch.equal(blens[1], blens[2])

    assert not torch.equal(lp__[0], lp__[1])
    assert not torch.equal(lp__[0], lp__[2])
    assert not torch.equal(lp__[1], lp__[2])


def test_draws_different_vi_project_up_nj():
    """Each draw from the sample should be different in likelihood."""
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs
    )
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )

    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(partials, weights, dim, embedder="up", connector="nj", soft_temp=1e-8)
    mymod.learn(epochs=2, path_write=None)

    # draw
    nsamples = 3
    _, blens, _, lp__ = mymod.draw_sample(nsamples, lp=True)
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
    mymod = DodonaphyVI(partials, weights, 2, embedder="up", connector="nj")
    tmp_dir = "./tmp_test_io"
    os.makedirs(tmp_dir, exist_ok=True)
    fp = os.path.join(tmp_dir, "test_data.csv")
    if os.path.exists(fp):
        os.remove(fp)
    mymod.save(fp)
    output = vi.read(fp, internals=False)
    os.remove(fp)
    os.rmdir(tmp_dir)
    assert allclose(
        output["leaf_mu"],
        mymod.VariationalParams["leaf_mu"].detach().numpy(),
        atol=1e-6,
    )
    assert allclose(
        output["leaf_sigma"],
        mymod.VariationalParams["leaf_sigma"].detach().numpy(),
        atol=1e-6,
    )
