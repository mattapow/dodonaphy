import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.laplace import Laplace
from dodonaphy.phylo import calculate_pairwise_distance, compress_alignment
import pytest


@pytest.mark.skip(reason="No way to ensure covariance matrix is positive definite yet.")
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
    hmap_inst = Laplace(
        partials[:],
        weights,
        dim=3,
        dists=dists,
        soft_temp=1e-6,
        loss_fn="likelihood",
        path_write=None,
        curvature=-1.0,
        prior="gammadir",
        tip_labels=None,
        matsumoto=False,
    )
    hmap_inst.learn(epochs=5, learn_rate=0.001, save_locations=False)
    hmap_inst.laplace(n_samples=2)
