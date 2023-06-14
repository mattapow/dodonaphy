import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import calculate_pairwise_distance, compress_alignment
import pytest
import numpy as np
import torch


@pytest.mark.parametrize(
    "model_name,loss_fn,prior,matsumoto,use_bito,connector",
    [
        ("JC69", "likelihood", "None", False, False, "nj"),
        ("JC69", "likelihood", "None", True, False, "nj"),
        ("JC69", "likelihood", "gammadir", False, False, "nj"),
        ("JC69", "likelihood", "uniform", False, False, "nj"),
        ("JC69", "likelihood", "normal", False, False, "nj"),
        ("JC69", "pair_likelihood", "None", False, False, "nj"),
        ("JC69", "hypHC", "None", False, False, "nj"),
        ("GTR", "likelihood", "None", False, False, "nj"),
        ("GTR", "likelihood", "None", True, False, "nj"),
        ("GTR", "likelihood", "gammadir", False, False, "nj"),
        ("GTR", "pair_likelihood", "None", False, False, "nj"),
        ("GTR", "hypHC", "None", False, False, "nj"),  # Sometimes errors? embedding locations are all nan.
        ("JC69", "likelihood", "None", False, True, "nj"),
        ("GTR", "likelihood", "None", False, True, "nj"),
        ("JC69", "likelihood", "None", False, False, "fix"),
    ],
)
def test_learn(model_name, loss_fn, prior, matsumoto, use_bito, connector):
    n_taxa = 8
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    msa_file = "test/tmp_learn.fasta"
    dna.write(path=msa_file, schema="fasta")
    partials, weights = compress_alignment(dna)

    if connector == "fix":
        peel = np.array([
            [2, 0, 8],
            [8, 4, 9],
            [3, 5, 10],
            [6, 10, 11],
            [1, 7, 12],
            [11, 12, 13],
            [9, 13, 14]
        ])
        n_nodes = 2*n_taxa - 1
        dists = np.random.exponential(scale=0.1, size=(n_nodes, n_nodes))
        dists = (dists + dists.transpose())/2.0
        np.fill_diagonal(dists, 0.0)
    else:
        peel = None
        dists = calculate_pairwise_distance(dna)

    hmap_inst = HMAP(
        partials[:],
        weights,
        dim=3,
        soft_temp=1e-6,
        loss_fn=loss_fn,
        path_write=None,
        curvature=-1.0,
        prior=prior,
        tip_labels=sim_tree.taxon_namespace.labels(),
        matsumoto=matsumoto,
        model_name=model_name,
        connector=connector,
        peel=peel,
    )
    location_file = None
    hmap_inst.init_embedding_params(location_file, dists, hydra_max_iter=0)
    if use_bito:
        peel, _ = hmap_inst.connect()
        hmap_inst.init_bito(msa_file, peel)
    hmap_inst.learn(epochs=2, learn_rate=0.001, save_locations=False)
    if model_name == "GTR":
        initial_value = torch.full([6], 1.0/6.0, dtype=torch.double)
        assert not torch.allclose(hmap_inst.phylomodel.sub_rates, initial_value), "Substitution rates unchanged"


def test_encode_decode():
    dna = dendropy.DnaCharacterMatrix.get(
        path="./test/data/ds1/dna.nex", schema="nexus"
    )
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna, adjust="JC69")
    hmap_inst = HMAP(
        partials[:],
        weights,
        dim=20,
        soft_temp=1e-6,
        loss_fn="likelihood",
        path_write=None,
        curvature=-1.0,
        prior="None",
        tip_labels=dna.taxon_namespace.labels(),
        matsumoto=False,
    )
    location_file = None
    hmap_inst.init_embedding_params(location_file, dists, hydra_max_iter=0)
    hmap_inst.learn(epochs=0, learn_rate=0.001, save_locations=False)

    # NJ tree from decenttree.
    # Then get the likelihood by loading into iq-tree and seeing the initial state.
    # use iqtree -s dna.nex -m JC -t dna.nj.newick --tree-fix
    log_likelihood_actual = -7130.968
    # Allow a tolerance since hydra+ is approximate
    tolerance = 30
    log_likelihood = -hmap_inst.loss.item()
    assert abs(log_likelihood - log_likelihood_actual) < tolerance
