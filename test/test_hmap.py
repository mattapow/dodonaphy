import dendropy
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import calculate_pairwise_distance, compress_alignment
import pytest


@pytest.mark.parametrize("model_name,loss_fn,prior,matsumoto", [
    ("JC69", "likelihood", "None", False),
    ("JC69", "likelihood", "None", True),
    ("JC69", "likelihood", "gammadir", False),
    ("JC69", "pair_likelihood", "None", False),
    ("JC69", "hypHC", "None", False),
    ("GTR", "likelihood", "None", False),
    ("GTR", "likelihood", "None", True),
    ("GTR", "likelihood", "gammadir", False),
    ("GTR", "pair_likelihood", "None", False),
    ("GTR", "hypHC", "None", False),
])
def test_learn(model_name, loss_fn, prior, matsumoto):
    n_taxa = 6
    sim_tree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=n_taxa
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=sim_tree, seq_model=dendropy.model.discrete.Jc69()
    )
    msa_file = "test/tmp_learn.fasta"
    dna.write(path=msa_file, schema="fasta")
    partials, weights = compress_alignment(dna)
    dists = calculate_pairwise_distance(dna)
    mymod = HMAP(
        partials[:],
        weights,
        dim=3,
        dists=dists,
        soft_temp=1e-6,
        loss_fn=loss_fn,
        path_write=None,
        msa_file=msa_file,
        curvature=-1.0,
        prior=prior,
        tip_labels=sim_tree.taxon_namespace.labels(),
        matsumoto=matsumoto,
        model_name=model_name
    )
    mymod.learn(epochs=2, learn_rate=0.001, save_locations=False)


def test_encode_decode():
    dna = dendropy.DnaCharacterMatrix.get(path="./test/data/ds1/dna.nex", schema="nexus")
    partials, weights = compress_alignment(dna)

    dists = calculate_pairwise_distance(dna, adjust="JC69")
    mymod = HMAP(
        partials[:],
        weights,
        dim=20,
        dists=dists,
        soft_temp=1e-6,
        loss_fn="likelihood",
        path_write=None,
        curvature=-1.0,
        prior="None",
        tip_labels=None,
        matsumoto=False,
    )
    mymod.learn(epochs=0, learn_rate=0.001, save_locations=False)

    # NJ tree from decenttree.
    # Then get the likelihood by loading into iq-tree and seeing the initial state.
    # use iqtree -s dna.nex -m JC -t dna.nj.newick --tree-fix
    log_likelihood_actual = -7130.968
    # Allow a tolerance since hydra+ is approximate
    tolerance = 30
    log_likelihood = - mymod.loss.item()
    assert abs(log_likelihood - log_likelihood_actual) < tolerance


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
    mymod = HMAP(
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
    mymod.learn(epochs=5, learn_rate=0.001, save_locations=False)
    mymod.laplace(n_samples=2)
