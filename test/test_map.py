import dendropy
import numpy as np
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy import utils
from dodonaphy.map import MAP
from dodonaphy.phylo import compress_alignment, calculate_pairwise_distance


def test_ml1():
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    dists = utils.tip_distances(sim_tree, n_taxa)
    mu = np.zeros(n_taxa)
    sigma = 0.01
    cov = np.ones_like(dists) * sigma
    threshold = min(dists[dists > 0])
    dists = dists + np.fmod(np.random.multivariate_normal(mu, cov, (6)), threshold)
    dists = (dists + dists.T) / 2
    mymod = MAP(partials[:], weights, dists=dists, soft_temp=1e-10, loss_fn="likelihood", prior="None")
    mymod.learn(epochs=1, learn_rate=1, path_write=None)


def test_map1():
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna)
    mymod = MAP(partials[:], weights, dists=dists, soft_temp=1e-10, loss_fn="likelihood", prior="gammadir")
    mymod.learn(epochs=1, learn_rate=1, path_write=None)