import bito
import bito.phylo_model_mapkeys as model_keys
import bito.phylo_gradient_mapkeys as gradient_keys
import dendropy
import torch
import numpy as np

from dodonaphy import tree as treeFunc
from dodonaphy import phylo, Cphylo

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
    blens_jacobian = np.array(jac_bito[0].gradient['branch_lengths'], copy=False)
    n_blen = 27*2 - 3
    n_blen_fake = 2
    assert len(blens_jacobian) == n_blen + n_blen_fake


def test_compute_likelihood_bito_file_gtr():
    msa_file = "./test/data/ds1/dna.fasta"
    tree_file = "./test/data/ds1/dna.nj.newick"
    bito_inst = bito.unrooted_instance("jc_as_gtr")
    bito_inst.read_fasta_file(msa_file)
    model_specification = bito.PhyloModelSpecification(substitution="GTR", site="constant", clock="strict")
    bito_inst.read_newick_file(tree_file)
    bito_inst.prepare_for_phylo_likelihood(model_specification, 1)
    phylo_model_param_block_map = bito_inst.get_phylo_model_param_block_map()
    phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_RATES][:] = np.repeat(1.0 / 6.0, 6)
    phylo_model_param_block_map[model_keys.SUBSTITUTION_MODEL_FREQUENCIES][:] = 0.25
    # compute the log likelihood and gradient
    ll = np.array(bito_inst.log_likelihoods())
    jac_bito = bito_inst.phylo_gradients()
    blens_jacobian = np.array(jac_bito[0].gradient['branch_lengths'], copy=False)
    model_jacobian = jac_bito[0].gradient['substitution_model']
    model_freq_jacobian = np.array(jac_bito[0].gradient['substitution_model_frequencies'], copy=False)
    model_rate_jacobian = np.array(jac_bito[0].gradient['substitution_model_rates'], copy=False)
    print(model_jacobian)
    n_blen = 27*2 - 3
    n_blen_fake = 2
    n_gtr = 6
    assert len(blens_jacobian) == n_blen + n_blen_fake
    assert len(model_freq_jacobian) == 3  # the four must sum to 1.0
    assert len(model_rate_jacobian) == 5  # TODO: only five as Q matrix is rescaled for unit substitution rate?


def test_compute_likelihood_bito_pb():
    msa_file = "./test/data/ds1/dna.fasta"
    msa_file_nex = "./test/data/ds1/dna.nex"
    tree_file = "./test/data/ds1/dna.nj.newick"
    # read into dendropy
    tree = dendropy.Tree.get(path=tree_file, schema="newick", preserve_underscores=True)
    # convert into dodonaphy (peel, blens)
    post_indexing, blens, name_id = treeFunc.dendrophy_to_pb(tree)
    # initisalise bito instance
    inst = bito.unrooted_instance("dodonaphy")
    inst.read_fasta_file(msa_file)  # read alignment
    # send to bito
    model_specification = bito.PhyloModelSpecification(substitution="JC69", site="constant", clock="strict")
    (bito_log_likelihood, jac) = phylo.calculate_treelikelihood_bito(inst, name_id, post_indexing, blens, model_specification)

    # compare to likelihood computed in dodonpahy
    dodo_log_likelihood = get_log_likelihood_dodonaphy(msa_file_nex, tree_file)
    assert np.isclose(bito_log_likelihood, dodo_log_likelihood)


def get_log_likelihood_dodonaphy(msa_file_nex, tree_file):
    dna = dendropy.DnaCharacterMatrix.get(path=msa_file_nex, schema="nexus")
    partials, weights, taxon_namespace = phylo.compress_alignment(dna, get_namespace=True)
    tree = dendropy.Tree.get(path=tree_file, schema="newick", taxon_namespace=taxon_namespace)
    post_indexing, blens, name_id = treeFunc.dendrophy_to_pb(tree)
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))
    mats = phylo.JC69_p_t(blens)
    freqs_np = np.full([4], 0.25)
    partials_np = [partial.detach().numpy() for partial in partials]
    weights_np = weights.detach().numpy()
    mats_np = mats.detach().numpy()
    dodo_log_likelihood = Cphylo.calculate_treelikelihood(
        partials_np, weights_np, post_indexing, mats_np, freqs_np
    )    
    return dodo_log_likelihood