import os
import pytest

import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import calculate_pairwise_distance, compress_alignment


def test_learn():
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna)
    mymod = HMAP(
        partials[:],
        weights,
        dim=3,
        dists=dists,
        soft_temp=1e-6,
        loss_fn="likelihood",
        curvature=-1.0,
        prior="None",
        tip_labels=None,
        matsumoto=False,
    )
    mymod.learn(epochs=2, learn_rate=0.001, path_write=None)


@pytest.mark.skip(reason="No way to covariance matrix is positive definite yet.")
def test_laplace():
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna)
    mymod = HMAP(
        partials[:],
        weights,
        dim=3,
        dists=dists,
        soft_temp=1e-6,
        loss_fn="likelihood",
        curvature=-1.0,
        prior="gammadir",
        tip_labels=None,
        matsumoto=False,
    )
    mymod.learn(epochs=5, learn_rate=0.001, path_write=None)
    mymod.laplace(path_write=None, n_samples=2)
