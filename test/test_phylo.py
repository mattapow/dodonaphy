import dendropy
import torch
import numpy as np
from dodonaphy import phylo, Cphylo
from dodonaphy import tree as treeFunc
from dodonaphy.base_model import BaseModel
from dodonaphy.phylomodel import PhyloModel
from pytest import approx
import pytest


def test_dna_alphabet():
    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nACGTRYMWSKBDHVN?-\n\n", schema="fasta"
    )

    partials, weights = phylo.compress_alignment(dna)
    sorted_partials = torch.tensor(
        [
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            [
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
            ],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(
        partials[0], sorted_partials
    ), "Partials incorrectly read alphabet."
    assert (sum(weights) / len(weights)) == 1, "Weights incorrectly read alphabet."


def test_calculate_pairwise_distance():
    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nAAAAAAAAA\n>T2\nAAAAAAAAD\n\n", schema="fasta"
    )
    dists = phylo.calculate_pairwise_distance(dna, adjust=None)
    true_dists = (np.ones(2) - np.eye(2)) / 9
    assert np.allclose(dists, true_dists)


def test_calculate_pairwise_distance_gaps():
    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\n-AAAAAAAA\n>T2\nAAAAAAAAD\n\n", schema="fasta"
    )
    dists = phylo.calculate_pairwise_distance(dna, adjust=None)
    true_dists = (np.ones(2) - np.eye(2)) * 1 / 9
    assert np.allclose(dists, true_dists)


def test_simple_dists_compare_decenttree_uncorrected():
    dna = dendropy.DnaCharacterMatrix.get(
        path="test/data/simple/dna.fasta", schema="fasta"
    )
    dists = phylo.calculate_pairwise_distance(dna)
    # compare to decenttree via
    # decenttree -fasta dna.fasta -out nj_uncorrected.newick -t NJ -dist-out NJ_dist_uncorrected.csv -uncorrected
    true_dists = np.genfromtxt(
        "test/data/simple/NJ_dist_uncorrected.csv", skip_header=1
    )
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])


def test_simple_dists_compare_decenttree_uncorrected_gaps():
    dna = dendropy.DnaCharacterMatrix.get(
        path="test/data/simple_gap/dna.fasta", schema="fasta"
    )
    dists = phylo.calculate_pairwise_distance(dna)
    # compare to decenttree via
    # decenttree -fasta dna.fasta -out nj_uncorrected.newick -t NJ -dist-out NJ_dist_uncorrected.csv -uncorrected
    true_dists = np.genfromtxt(
        "test/data/simple_gap/NJ_dist_uncorrected.csv", skip_header=1
    )
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])


def test_simple_dists_compare_decenttree():
    dna = dendropy.DnaCharacterMatrix.get(
        path="test/data/simple/dna.fasta", schema="fasta"
    )
    dists = phylo.calculate_pairwise_distance(dna, adjust="JC69")
    # compare to decenttree via
    # decenttree -fasta dna.fasta -out nj.newick -t NJ -dist-out NJ_dist.csv
    true_dists = np.genfromtxt("test/data/simple/NJ_dist.csv", skip_header=1)
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])


def test_likelihood_alphabet():
    blen = torch.ones(3, dtype=torch.double) / 10
    post_indexing = [[0, 1, 2]]
    mats = PhyloModel.JC69_p_t(blen)
    freqs = torch.full([4], 0.25, dtype=torch.float64)

    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nAAAAAAAAA\n>T2\nAAAAAAAAD\n\n", schema="fasta"
    )
    partials, weights = phylo.compress_alignment(dna)
    partials.append(torch.zeros((1, 4, 9), dtype=torch.float64))
    LL = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)

    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nAAAAAAAAA\n>T2\nAAAAAAAA-\n\n", schema="fasta"
    )
    partials, weights = phylo.compress_alignment(dna)
    partials.append(torch.zeros((1, 4, 9), dtype=torch.float64))
    LL_1 = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)
    assert LL != LL_1, "Likelihoods should be different."


def test_prior_mrbayes_0():
    data = "(6:2.000000e-02,((5:2.000000e-02,2:2.000000e-02):2.000000e-02,\
        (4:2.000000e-02,3:2.000000e-02):2.000000e-02):2.000000e-02,1:2.000000e-02);"
    dendo_tree = dendropy.Tree.get(data=data, schema="newick")
    _, blens, _ = treeFunc.dendropy_to_pb(dendo_tree)

    lnPrior = BaseModel.compute_prior_gamma_dir(
        blens,
        aT=torch.ones(1),
        bT=torch.full((1,), 0.1),
        a=torch.ones(1),
        c=torch.ones(1),
    )
    assert lnPrior == approx(17.34844, abs=1e-5)


def test_prior_mrbayes_1():
    data = "((2:1.179257e-01,(6:1.047047e-03,3:1.426334e-03):7.713732e-02):\
        7.894008e-02,(4:2.848264e-03,5:2.711671e-03):1.879317e-03,1:4.419083e-03);"
    dendo_tree = dendropy.Tree.get(data=data, schema="newick")
    _, blens, _ = treeFunc.dendropy_to_pb(dendo_tree)
    lnPrior = BaseModel.compute_prior_gamma_dir(
        blens,
        aT=torch.ones(1),
        bT=torch.full((1,), 0.1),
        a=torch.ones(1),
        c=torch.ones(1),
    )
    assert lnPrior == approx(13.56829, abs=1e-5)


def test_likelihood_mrbayes_torch():
    dna = dendropy.DnaCharacterMatrix.get(
        path="./test/data/ds1/dna.nex", schema="nexus"
    )
    partials, weights, taxon_namespace = phylo.compress_alignment(
        dna, get_namespace=True
    )

    """ "Tree and likelihood copied from MrBayes output."""
    tree = dendropy.Tree.get(
        path="./test/data/ds1/mb_tree300000.nex",
        schema="nexus",
        taxon_namespace=taxon_namespace,
    )
    post_indexing, blens, name_id = treeFunc.dendropy_to_pb(tree)

    # append space for internal node partials
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))

    mats = PhyloModel.JC69_p_t(blens)
    freqs = torch.full([4], 0.25, dtype=torch.float64)

    ln_p = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)
    assert ln_p == approx(-6904.632, abs=1e-3)


def test_likelihood_mrbayes_numpy():
    dna = dendropy.DnaCharacterMatrix.get(
        path="./test/data/ds1/dna.nex", schema="nexus"
    )
    partials, weights, taxon_namespace = phylo.compress_alignment(
        dna, get_namespace=True
    )

    # example tree
    tree = dendropy.Tree.get(
        path="./test/data/ds1/mb_tree300000.nex",
        schema="nexus",
        taxon_namespace=taxon_namespace,
    )
    post_indexing, blens, name_id = treeFunc.dendropy_to_pb(tree)

    # append space for internal node partials
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))

    mats = PhyloModel.JC69_p_t(blens)
    freqs_np = np.full([4], 0.25)

    partials_np = [partial.detach().numpy() for partial in partials]
    weights_np = weights.detach().numpy()
    mats_np = mats.detach().numpy()
    ln_p = Cphylo.calculate_treelikelihood(
        partials_np, weights_np, post_indexing, mats_np, freqs_np
    )

    assert ln_p == approx(-6904.632, abs=1e-3)


def test_GTR_equals_JC69():
    blens = torch.tensor(np.array([0.1]), dtype=torch.double)
    mats_JC = PhyloModel.JC69_p_t(blens)

    sub_rates = torch.full([6], 1.0, dtype=torch.double)
    freqs = torch.full([4], 0.25, dtype=torch.double)
    mats_GTR = PhyloModel.GTR_p_t(blens, sub_rates, freqs)

    assert torch.allclose(mats_JC, mats_GTR)


@pytest.mark.parametrize(
    "blens_in,size",
    [
        ([[1.0, 1.0]], [1, 2, 4, 4]),
        ([[1.0]], [1, 1, 4, 4]),
        ([[1.0], [1.0]], [2, 1, 4, 4]),
        ([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [3, 2, 4, 4]),
    ],
)
def test_GTR_mats_size(blens_in, size):
    sub_rates = torch.full([6], 1.0, dtype=torch.double)
    freqs = torch.full([4], 0.25, dtype=torch.double)
    blens = torch.tensor(np.array(blens_in), dtype=torch.double)
    mats_GTR = PhyloModel.GTR_p_t(blens, sub_rates, freqs)
    assert mats_GTR.shape == torch.Size(size)
