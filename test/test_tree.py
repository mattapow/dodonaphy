import dendropy
import numpy as np
import torch
from dendropy import Tree

from dodonaphy import tree as treeFunc


def test_tree_to_newick_empty():
    # Empty tree
    name_id = {}
    peel = []
    blens = torch.Tensor([])
    rooted = True
    expected = ";"
    assert treeFunc.tree_to_newick(name_id, peel, blens, rooted) == expected


def test_dendropy_to_pb_simple():
    data = "(bird:6.000000e-02,((chimp:5.000000e-02,dog:2.000000e-02):17.000000e-02,\
        (crocodile:4.000000e-02,emu:3.000000e-02):18.000000e-02):19.000000e-02,kangaroo:1.000000e-02);"
    dendo_tree = Tree.get(data=data, schema="newick")
    peel, blens, name_id = treeFunc.dendropy_to_pb(dendo_tree)

    # compare to hard code
    true_peel = np.array([[1, 2, 6], [3, 4, 7], [6, 7, 8], [0, 8, 9], [5, 9, 10]])
    assert np.allclose(peel, true_peel)
    true_blens = torch.tensor(
        [
            0.0600,
            0.0500,
            0.0200,
            0.0400,
            0.0300,
            0.0100,
            0.1700,
            0.1800,
            0.1900,
            0.0000,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(blens, true_blens)


def test_dendropy_to_pb_and_back_ds1():
    taxa = dendropy.TaxonNamespace()

    tree_file = "./test/data/ds1/dna.nj.newick"
    dendro_tree = dendropy.Tree.get(
        path=tree_file,
        schema="newick",
        rooting="force-unrooted",
        taxon_namespace=taxa,
        preserve_underscores=True,
    )

    post_indexing, blens, name_id = treeFunc.dendropy_to_pb(dendro_tree)

    # compare to putting back into newick and then dendropy
    nwk = treeFunc.tree_to_newick(name_id, post_indexing, blens)
    dendro_tree_dodo = dendropy.Tree.get(
        string=nwk,
        schema="newick",
        rooting="force-unrooted",
        taxon_namespace=taxa,
        preserve_underscores=True,
    )
    diff = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(
        dendro_tree, dendro_tree_dodo
    )

    assert np.isclose(diff, 0.0, atol=1e-4)


def test_tree_to_newick_unrooted():
    data = "((bird:0.06,((chimp:0.05,dog:0.02):0.17,(crocodile:0.04,emu:0.03):0.18):0.19):1.0,kangaroo:0.01);"
    dendo_tree = Tree.get(data=data, schema="newick", rooting="force-unrooted")
    dendo_tree.deroot()
    peel, blens, name_id = treeFunc.dendropy_to_pb(dendo_tree)
    # compare to hard code
    true_peel = np.array([[1, 2, 6], [3, 4, 7], [6, 7, 8], [0, 8, 9], [5, 9, 10]])
    assert np.allclose(peel, true_peel)
    true_blens = torch.tensor(
        [
            0.0600,
            0.0500,
            0.0200,
            0.0400,
            0.0300,
            1.0100,
            0.1700,
            0.1800,
            0.1900,
            0.0000,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(blens, true_blens)

    # compare to putting back into newick and then dendropy
    # needs to be rooted to be read in by dendropy
    nwk = treeFunc.tree_to_newick(name_id, peel, blens, rooted=True)
    dendro_tree_dodo = dendropy.Tree.get(
        string=nwk, schema="newick", taxon_namespace=dendo_tree.taxon_namespace
    )
    diff = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(
        dendo_tree, dendro_tree_dodo
    )
    assert np.isclose(diff, 0.0)


def test_tree_to_newick_rooted():
    peel = np.array([[1, 2, 6], [3, 4, 7], [6, 7, 8], [0, 8, 9], [5, 9, 10]])
    blens = torch.tensor(
        [
            0.0600,
            0.0500,
            0.0200,
            0.0400,
            0.0300,
            1.0100,
            0.1700,
            0.1800,
            0.1900,
            0.0000,
        ],
        dtype=torch.float64,
    )
    taxon_names = ["a", "b", "c", "d", "e", "f"]
    name_dict = {name: id for id, name in enumerate(taxon_names)}
    nwk = treeFunc.tree_to_newick(name_dict, peel, blens, rooted=True)

    tree = Tree.get(data=nwk, schema="newick")
    assert len(tree.leaf_nodes()) == len(taxon_names)

    for leaf in tree.leaf_nodes():
        assert leaf.taxon.label in taxon_names

    for edge in tree.postorder_edge_iter():
        assert edge.length is None or edge.length >= 0.0

    expected_nwk = "(f:1.01,(a:0.06,((b:0.05,c:0.02):0.17,(d:0.04,e:0.03):0.18):0.19):0.00);"

    expected_tree = dendropy.Tree.get(string=expected_nwk, schema="newick")
    tree = dendropy.Tree.get(
        string=nwk, schema="newick", taxon_namespace=expected_tree.taxon_namespace
    )
    diff = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(
        tree, expected_tree
    )
    assert np.isclose(diff, 0.0)
