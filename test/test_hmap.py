import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import compress_alignment, calculate_pairwise_distance


def test_hmap():
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna)
    mymod = HMAP(partials[:],
        weights,
        dim=3,
        dists=dists,
        soft_temp=1e-6,
        loss_fn="likelihood",
        curvature=-1.,
        prior="None",
        tip_labels=None,
        matsumoto=False,)
    import torch
    with torch.autograd.detect_anomaly():
        mymod.learn(epochs=20, learn_rate=0.001, path_write=None)
