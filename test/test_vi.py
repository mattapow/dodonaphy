import os

import dendropy
import numpy as np
import pytest
import torch
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from hydraPlus import hydraPlus
from dodonaphy import vi
from dodonaphy.phylo import calculate_pairwise_distance, compress_alignment
from dodonaphy.vi import DodonaphyVI
from numpy import allclose
from dodonaphy.phylomodel import PhyloModel


@pytest.mark.parametrize("embedder,connector",
                         [("up", "geodesics"),
                          ("up", "nj"),
                          ("wrap", "nj"),
                          ("wrap", "geodesics")])
def test_draws_different_vi(embedder, connector):
    """Each draw from the sample should be different in likelihood."""
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=6
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(
        partials, weights, dim=3, embedder=embedder, connector=connector, soft_temp=1e-8
    )
    mix_weights = np.ones((1))
    leaf_loc_hyp = np.random.randn(6, 3)
    leaf_sigma = np.abs(leaf_loc_hyp) * 0.01
    param_init = {
        "leaf_mu": torch.from_numpy(leaf_loc_hyp).double(),
        "leaf_sigma": torch.from_numpy(leaf_sigma).double(),
        "mix_weights": torch.tensor(mix_weights, dtype=torch.float64),
    }
    mymod.set_variationalParams(param_init)
    mymod.learn(epochs=2, path_write=None, importance_samples=3)

    _, blens, _, log_like, _ = mymod.draw_sample(3, get_likelihood=True)
    assert not torch.equal(blens[0], blens[1])
    assert not torch.equal(blens[0], blens[2])
    assert not torch.equal(blens[1], blens[2])

    assert not torch.equal(log_like[0], log_like[1])
    assert not torch.equal(log_like[0], log_like[2])
    assert not torch.equal(log_like[1], log_like[2])


@pytest.mark.parametrize(
    "model_name",
    ("JC69", "GTR"),
)
def test_models_ds1(model_name):
    dna = dendropy.DnaCharacterMatrix.get(path="./test/data/ds1/dna.nex", schema="nexus")
    partials, weights = compress_alignment(dna)
    dim = 3
    mymod = DodonaphyVI(
        partials, weights, dim, embedder="up", connector="nj", soft_temp=1e-8, model_name=model_name
    )
    dists = calculate_pairwise_distance(dna, adjust="JC69")
    hp_obj = hydraPlus.HydraPlus(dists, dim=dim, curvature=-1.0)
    emm_tips = hp_obj.embed(equi_adj=0.0, alpha=1.1)
    coef_var = 0.1
    leaf_sigma = emm_tips["X"] * coef_var
    mix_weights = np.ones((1))
    param_init = {
        "leaf_mu": torch.tensor(
            emm_tips["X"], dtype=torch.float64
        ),
        "leaf_sigma": torch.tensor(
            leaf_sigma, dtype=torch.float64
        ),
        "mix_weights": torch.tensor(mix_weights, dtype=torch.float64),
    }
    phylomodel = PhyloModel(model_name)
    if not phylomodel.fix_sub_rates:
        param_init["sub_rates"] = phylomodel.sub_rates
    if not phylomodel.fix_freqs:
        param_init["freqs"] = phylomodel.freqs
    mymod.set_variationalParams(param_init)
    mymod.learn(epochs=2, path_write=None, importance_samples=1)


@pytest.mark.skip(reason="We don't have to read this in. Would need to fix the read function.")
def test_vi_io():
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=6
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(partials, weights, 2, embedder="up", connector="nj")
    tmp_dir = "./test/tmp_test_io"
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


@pytest.mark.parametrize("model_name", ("JC69", "GTR"))
def test_bito_ds1(model_name):
    dna_nex = "./test/data/ds1/dna.nex"
    dna_fasta = "./test/data/ds1/dna.fasta"
    dna = dendropy.DnaCharacterMatrix.get(path=dna_nex, schema="nexus")
    partials, weights = compress_alignment(dna)
    dim = 3
    mymod = DodonaphyVI(
        partials, weights, dim, embedder="up", connector="nj", soft_temp=1e-8, model_name=model_name
    )
    dists = calculate_pairwise_distance(dna, adjust="JC69")
    hp_obj = hydraPlus.HydraPlus(dists, dim=dim, curvature=-1.0)
    emm_tips = hp_obj.embed(equi_adj=0.0, alpha=1.1)
    coef_var = 0.1
    leaf_sigma = emm_tips["X"] * coef_var
    mix_weights = np.ones((1))
    param_init = {
        "leaf_mu": torch.tensor(
            emm_tips["X"], dtype=torch.float64
        ),
        "leaf_sigma": torch.tensor(
            leaf_sigma, dtype=torch.float64
        ),
        "mix_weights": torch.tensor(mix_weights, dtype=torch.float64),
    }
    phylomodel = PhyloModel("JC69")
    if not phylomodel.fix_sub_rates:
        param_init["sub_rates"] = phylomodel.sub_rates
    if not phylomodel.fix_freqs:
        param_init["freqs"] = phylomodel.freqs
    mymod.set_variationalParams(param_init)

    # initialise bito using a sequence alignment and a (postorder) tree
    peel, _, _ = mymod.connect(param_init["leaf_mu"])
    mymod.init_bito(dna_fasta, peel)
    mymod.learn(epochs=2, path_write=None, importance_samples=1)
