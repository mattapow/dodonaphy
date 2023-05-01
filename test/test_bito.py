import bito
import bito.phylo_model_mapkeys as model_keys
import dendropy
import numpy as np
import pytest
import torch

from dodonaphy import Cphylo, phylo
from dodonaphy.phylomodel import PhyloModel
from dodonaphy import tree as treeFunc


def test_compute_likelihood_bito_file():
    msa_file = "./test/data/ds1/dna.fasta"
    tree_file = "./test/data/ds1/dna.nj.newick"
    bito_inst = bito.unrooted_instance("simple_jc")
    bito_inst.read_fasta_file(msa_file)
    bito_inst.read_newick_file(tree_file)
    model_specification = bito.PhyloModelSpecification(
        substitution="JC69", site="constant", clock="strict"
    )
    bito_inst.prepare_for_phylo_likelihood(model_specification, 1)
    ll = np.array(bito_inst.log_likelihoods())
    jac_bito = bito_inst.phylo_gradients()
    blens_jacobian = np.array(jac_bito[0].gradient["branch_lengths"], copy=False)
    n_blen = 27 * 2 - 3
    n_blen_fake = 2
    assert len(blens_jacobian) == n_blen + n_blen_fake
    # comparing to value from first run
    assert np.allclose(ll, -7130.96813417)


def test_compute_likelihood_bito_file_gtr():
    msa_file = "./test/data/ds1/dna.fasta"
    tree_file = "./test/data/ds1/dna.nj.newick"
    bito_inst = bito.unrooted_instance("jc_as_gtr")
    bito_inst.read_fasta_file(msa_file)
    model_specification = bito.PhyloModelSpecification(
        substitution="GTR", site="constant", clock="strict"
    )
    bito_inst.read_newick_file(tree_file)
    bito_inst.prepare_for_phylo_likelihood(model_specification, 1)
    phylo_model_param_block_map = bito_inst.get_phylo_model_param_block_map()
    phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_RATES][:] = np.repeat(
        1.0 / 6.0, 6
    )
    phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_FREQUENCIES][:] = 0.25
    # compute the log likelihood and gradient
    ll = np.array(bito_inst.log_likelihoods())
    assert not np.isnan(ll)
    jac_bito = bito_inst.phylo_gradients()
    blens_grad = np.array(jac_bito[0].gradient["branch_lengths"], copy=False)
    model_grad = jac_bito[0].gradient["substitution_model"]
    model_freq_grad = np.array(
        jac_bito[0].gradient["substitution_model_frequencies"], copy=False
    )
    model_rate_grad = np.array(
        jac_bito[0].gradient["substitution_model_rates"], copy=False
    )
    print(model_grad)
    n_blen = 27 * 2 - 3
    n_blen_fake = 2
    n_gtr = 6
    assert len(blens_grad) == n_blen + n_blen_fake
    assert len(model_freq_grad) == 3  # the four must sum to 1.0
    assert len(model_rate_grad) == n_gtr - 1


def get_log_likelihood_dodonaphy(msa_file_nex, tree_file):
    dna = dendropy.DnaCharacterMatrix.get(path=msa_file_nex, schema="nexus")
    partials, weights, taxon_namespace = phylo.compress_alignment(
        dna, get_namespace=True
    )
    tree = dendropy.Tree.get(
        path=tree_file, schema="newick", taxon_namespace=taxon_namespace
    )
    post_indexing, blens, name_id = treeFunc.dendropy_to_pb(tree)
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))
    mats = PhyloModel.JC69_p_t(blens)
    freqs_np = np.full([4], 0.25)
    partials_np = [partial.detach().numpy() for partial in partials]
    weights_np = weights.detach().numpy()
    mats_np = mats.detach().numpy()
    dodo_log_likelihood = Cphylo.calculate_treelikelihood(
        partials_np, weights_np, post_indexing, mats_np, freqs_np
    )
    return dodo_log_likelihood


@pytest.mark.parametrize(
    "model_name, torch_mode",
    (
        ["JC69", "forward"],
        ["JC69", "backward"],
        ["GTR", "forward"],
        ["GTR", "backward"],
    ),
)
def test_compute_LL_bito(torch_mode, model_name):
    # read tree into dendropy
    tree_file = "./test/data/ds1/dna.nj.newick"
    tree = dendropy.Tree.get(path=tree_file, schema="newick", preserve_underscores=True)
    # convert into dodonaphy (peel, blens)
    peel, blens, name_id = treeFunc.dendropy_to_pb(tree)

    bito_inst = bito.unrooted_instance("testing")
    # read msa
    msa_file = "./test/data/ds1/dna.fasta"
    bito_inst.read_fasta_file(msa_file)
    # specify model
    model_specification = bito.PhyloModelSpecification(
        substitution=model_name, site="constant", clock="strict"
    )
    sub_rates = torch.zeros((5))
    freqs = torch.zeros((3))
    # need to load trees into inst before prepare_for_phylo_likelihood
    # just read in the original tree even though we'll use the peel and blens
    bito_inst.read_newick_file(tree_file)
    bito_inst.prepare_for_phylo_likelihood(model_specification, 1)

    if torch_mode == "forward":
        # compute likelihood with bito
        bito_log_likelihood = phylo.TreeLikelihood.apply(
            blens, peel, bito_inst, sub_rates, freqs
        )
        # compare to calculation without bito
        msa_file_nex = "./test/data/ds1/dna.nex"
        dodo_log_likelihood = get_log_likelihood_dodonaphy(msa_file_nex, tree_file)
        assert pytest.approx(bito_log_likelihood.item()) == dodo_log_likelihood
    elif torch_mode == "backward":
        # initialise optimisation object
        params = {"blens": blens.clone().detach().requires_grad_()}
        optimizer = torch.optim.Adam(params=list(params.values()), lr=0.01)
        # optimse the branch lengths
        optimizer.zero_grad()
        loss = phylo.TreeLikelihood.apply(
            params["blens"], peel, bito_inst, sub_rates, freqs
        )
        loss.backward()
        optimizer.step()


@pytest.mark.parametrize(
    "model_name, torch_mode",
    (
        ["JC69", "forward"],
        ["JC69", "backward"],
        ["GTR", "forward"],
        ["GTR", "backward"],
    ),
)
def test_compute_LL_bito_ofParentID(torch_mode, model_name):
    # read tree into dendropy
    tree_file = "./test/data/ds1/dna.nj.newick"
    tree = dendropy.Tree.get(path=tree_file, schema="newick", preserve_underscores=True)
    # convert into dodonaphy (peel, blens)
    peel, blens, _ = treeFunc.dendropy_to_pb(tree)

    bito_inst = bito.unrooted_instance("testing")
    # read msa
    msa_file = "./test/data/ds1/dna.fasta"
    bito_inst.read_fasta_file(msa_file)
    # specify model
    model_specification = bito.PhyloModelSpecification(
        substitution=model_name, site="constant", clock="strict"
    )
    sub_rates = torch.zeros((5))
    freqs = torch.zeros((3))

    parent_id = phylo.get_parent_id_vector(peel, rooted=False)
    tree = bito.UnrootedTree.of_parent_id_vector(parent_id)
    bito_inst.tree_collection = bito.UnrootedTreeCollection([tree])
    bito_inst.prepare_for_phylo_likelihood(model_specification, 1)

    if torch_mode == "forward":
        # compute likelihood with bito
        bito_log_likelihood = phylo.TreeLikelihood.apply(
            blens, peel, bito_inst, sub_rates, freqs
        )
        # compare to calculation without bito
        msa_file_nex = "./test/data/ds1/dna.nex"
        dodo_log_likelihood = get_log_likelihood_dodonaphy(msa_file_nex, tree_file)
        assert pytest.approx(bito_log_likelihood.item()) == dodo_log_likelihood
    elif torch_mode == "backward":
        # initialise optimisation object
        params = {"blens": blens.clone().detach().requires_grad_()}
        optimizer = torch.optim.Adam(params=list(params.values()), lr=0.01)
        # optimse the branch lengths
        optimizer.zero_grad()
        loss = phylo.TreeLikelihood.apply(
            params["blens"], peel, bito_inst, sub_rates, freqs
        )
        loss.backward()
        optimizer.step()
