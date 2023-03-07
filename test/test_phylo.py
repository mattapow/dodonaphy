import dendropy
import torch
import numpy as np
from dodonaphy import phylo, Cphylo
from dodonaphy import tree as treeFunc
from dodonaphy.base_model import BaseModel
from pytest import approx


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
    dna = dendropy.DnaCharacterMatrix.get(path="test/data/simple/dna.fasta", schema="fasta")
    dists = phylo.calculate_pairwise_distance(dna)
    # compare to decenttree via 
    # decenttree -fasta dna.fasta -out nj_uncorrected.newick -t NJ -dist-out NJ_dist_uncorrected.csv -uncorrected
    true_dists = np.genfromtxt("test/data/simple/NJ_dist_uncorrected.csv", skip_header=1)
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])

def test_simple_dists_compare_decenttree_uncorrected_gaps():
    dna = dendropy.DnaCharacterMatrix.get(path="test/data/simple_gap/dna.fasta", schema="fasta")
    dists = phylo.calculate_pairwise_distance(dna)
    # compare to decenttree via 
    # decenttree -fasta dna.fasta -out nj_uncorrected.newick -t NJ -dist-out NJ_dist_uncorrected.csv -uncorrected
    true_dists = np.genfromtxt("test/data/simple_gap/NJ_dist_uncorrected.csv", skip_header=1)
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])

def test_simple_dists_compare_decenttree():
    dna = dendropy.DnaCharacterMatrix.get(path="test/data/simple/dna.fasta", schema="fasta")
    dists = phylo.calculate_pairwise_distance(dna, adjust="JC69")
    # compare to decenttree via 
    # decenttree -fasta dna.fasta -out nj.newick -t NJ -dist-out NJ_dist.csv
    true_dists = np.genfromtxt("test/data/simple/NJ_dist.csv", skip_header=1)
    true_dists[np.isnan(true_dists)] = 0
    assert np.allclose(dists, true_dists[:, 1:])

def test_likelihood_alphabet():
    blen = torch.ones(3, dtype=torch.double) / 10
    post_indexing = [[0, 1, 2]]
    mats = phylo.JC69_p_t(blen)
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


def test_likelihood_mrbayes_torch():
    """ "Tree and likelihood copied from MrBayes output."""
    data = "(24:4.561809e-03,((22:4.904862e-03,10:9.550327e-03):5.704741e-03,(((15:1.985269e-02,(((((2:5.486892e-03,23:9.486339e-03):3.878666e-03,26:7.583491e-03):6.373763e-04,(5:1.661778e-02,((21:3.649255e-03,19:4.450261e-03):4.958361e-03,(((9:7.469533e-04,3:1.707387e-02):2.598443e-03,13:5.465068e-03):3.836386e-03,14:1.029500e-02):3.334175e-03):3.110589e-03):5.036948e-03):6.302069e-03,((4:1.227562e-02,12:1.724676e-02):2.311179e-03,((6:1.464740e-02,17:1.531754e-02):8.259293e-03,8:2.203792e-02):8.089750e-03):2.968914e-03):9.012311e-03,27:3.346403e-03):1.134159e-02):1.279327e-02,((11:1.221072e-03,(20:6.189060e-03,16:4.987891e-03):1.805937e-03):4.928884e-03,18:3.861566e-03):2.671721e-02):9.188388e-03,(25:9.749763e-03,7:7.646632e-03):1.097835e-02):8.034396e-03):1.869811e-03,1:1.815073e-03);"
    tree = dendropy.Tree.get(data=data, schema="newick")
    post_indexing, blens = treeFunc.dendrophy_to_pb(tree, offset=1)

    dna = dendropy.DnaCharacterMatrix.get(path="./test/data/ds1/dna.nex", schema="nexus")
    partials, weights = phylo.compress_alignment(dna)
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))

    mats = phylo.JC69_p_t(blens)
    freqs = torch.full([4], 0.25, dtype=torch.float64)

    ln_p = phylo.calculate_treelikelihood(partials, weights, post_indexing, mats, freqs)
    assert ln_p == approx(-6904.632, abs=1e-3)


def test_likelihood_mrbayes_numpy():
    data = "(24:4.561809e-03,((22:4.904862e-03,10:9.550327e-03):5.704741e-03,(((15:1.985269e-02,(((((2:5.486892e-03,23:9.486339e-03):3.878666e-03,26:7.583491e-03):6.373763e-04,(5:1.661778e-02,((21:3.649255e-03,19:4.450261e-03):4.958361e-03,(((9:7.469533e-04,3:1.707387e-02):2.598443e-03,13:5.465068e-03):3.836386e-03,14:1.029500e-02):3.334175e-03):3.110589e-03):5.036948e-03):6.302069e-03,((4:1.227562e-02,12:1.724676e-02):2.311179e-03,((6:1.464740e-02,17:1.531754e-02):8.259293e-03,8:2.203792e-02):8.089750e-03):2.968914e-03):9.012311e-03,27:3.346403e-03):1.134159e-02):1.279327e-02,((11:1.221072e-03,(20:6.189060e-03,16:4.987891e-03):1.805937e-03):4.928884e-03,18:3.861566e-03):2.671721e-02):9.188388e-03,(25:9.749763e-03,7:7.646632e-03):1.097835e-02):8.034396e-03):1.869811e-03,1:1.815073e-03);"
    tree = dendropy.Tree.get(data=data, schema="newick")
    post_indexing, blens = treeFunc.dendrophy_to_pb(tree, offset=1)

    dna = dendropy.DnaCharacterMatrix.get(path="./test/data/ds1/dna.nex", schema="nexus")
    partials, weights = phylo.compress_alignment(dna)
    L = partials[0].shape[1]
    for _ in range(27 - 1):
        partials.append(torch.zeros((1, 4, L), dtype=torch.float64))

    mats = phylo.JC69_p_t(blens)
    freqs_np = np.full([4], 0.25)

    partials_np = [partial.detach().numpy() for partial in partials]
    weights_np = weights.detach().numpy()
    mats_np = mats.detach().numpy()
    ln_p = Cphylo.calculate_treelikelihood(
        partials_np, weights_np, post_indexing, mats_np, freqs_np
    )

    assert ln_p == approx(-6904.632, abs=1e-3)
