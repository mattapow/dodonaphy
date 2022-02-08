import dendropy
import torch
from dodonaphy import phylo
from dodonaphy import tree as treeFunc
from dodonaphy.base_model import BaseModel
from pytest import approx
import numpy as np


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
    assert np.allclose(
        partials[0], sorted_partials
    ), "Partials incorrectly read alphabet."
    assert (sum(weights) / len(weights)) == 1, "Weights incorrectly read alphabet."


def test_likelihood_alphabet():
    blen = torch.ones(3, dtype=torch.double) / 10
    post_indexing = [[0, 1, 2]]
    mats = phylo.JC69_p_t(blen)
    freqs = torch.full([4], 0.25, dtype=torch.float64)

    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nAAAAAAAAA\n>T2\nAAAAAAAAD\n\n", schema="fasta"
    )
    partials_np, weights = phylo.compress_alignment(dna)
    partials_np.append(np.zeros((1, 4, 9), dtype=np.float64))
    partials = [torch.from_numpy(plv) for plv in partials_np]
    LL = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)

    dna = dendropy.DnaCharacterMatrix.get(
        data=">T1\nAAAAAAAAA\n>T2\nAAAAAAAA-\n\n", schema="fasta"
    )
    partials_np, weights = phylo.compress_alignment(dna)
    partials_np.append(np.zeros((1, 4, 9), dtype=np.float64))
    partials = [torch.from_numpy(plv) for plv in partials_np]
    LL_1 = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)
    assert LL != LL_1, "Likelihoods should be different."

def test_prior_mrbayes_0():
    data = "(6:2.000000e-02,((5:2.000000e-02,2:2.000000e-02):2.000000e-02,\
        (4:2.000000e-02,3:2.000000e-02):2.000000e-02):2.000000e-02,1:2.000000e-02);"
    dendo_tree = dendropy.Tree.get(data=data, schema="newick")
    _, blens = treeFunc.dendrophy_to_pb(dendo_tree)

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
    _, blens = treeFunc.dendrophy_to_pb(dendo_tree)

    lnPrior = BaseModel.compute_prior_gamma_dir(
        blens,
        aT=torch.ones(1),
        bT=torch.full((1,), 0.1),
        a=torch.ones(1),
        c=torch.ones(1),
    )
    assert lnPrior == approx(13.56829, abs=1e-5)
