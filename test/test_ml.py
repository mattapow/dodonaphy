import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy import utils
from dodonaphy.ml import ML
from dodonaphy.phylo import compress_alignment


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
    ML.run(
        n_taxa,
        partials,
        weights,
        dists,
        path_write=None,
        epochs=10,
        lr=1,
        soft_temp=1e-4
    )
