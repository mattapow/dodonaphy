from dendropy import Tree
import dendropy
import torch
import numpy as np

from dodonaphy import tree as treeFunc


def test_dendropy_to_pb_simple():
    data = "(bird:6.000000e-02,((chimp:5.000000e-02,dog:2.000000e-02):17.000000e-02,\
        (crocodile:4.000000e-02,emu:3.000000e-02):18.000000e-02):19.000000e-02,kangaroo:1.000000e-02);"
    dendo_tree = Tree.get(data=data, schema="newick")
    dendo_tree.print_plot()
    peel, blens, name_id = treeFunc.dendropy_to_pb(dendo_tree)

    # compare to hard code
    true_peel = np.array(
      [[ 1,  2,  6],
       [ 3,  4,  7],
       [ 6,  7,  8],
       [ 0,  8,  9],
       [ 5,  9, 10]])
    assert np.allclose(peel, true_peel)
    true_blens = torch.tensor([0.0600, 0.0500, 0.0200, 0.0400, 0.0300, 0.0100, 0.1700, 0.1800, 0.1900,
        0.0000], dtype=torch.float64)
    assert torch.allclose(blens, true_blens)

    # compare to putting back into newick and then dendropy
    nwk = treeFunc.tree_to_newick(name_id, peel, blens)
    dendro_tree_dodo = dendropy.Tree.get(string=nwk, schema="newick", taxon_namespace=dendo_tree.taxon_namespace)
    diff = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(dendo_tree, dendro_tree_dodo)
    assert np.isclose(diff, 0.0)


def test_dendropy_to_pb_ds1():
    taxa = dendropy.TaxonNamespace()

    tree_file = "./test/data/ds1/dna.nj.newick"
    dendro_tree = dendropy.Tree.get(
        path=tree_file,
        schema="newick",
        rooting="force-unrooted",
        taxon_namespace=taxa,
        preserve_underscores=True)
    post_indexing, blens, name_id = treeFunc.dendropy_to_pb(dendro_tree)

    # compare to putting back into newick and then dendropy
    nwk = treeFunc.tree_to_newick(name_id, post_indexing, blens)
    dendro_tree_dodo = dendropy.Tree.get(
        string=nwk,
        schema="newick",
        rooting="force-unrooted",
        taxon_namespace=taxa,
        preserve_underscores=True)
    diff = dendropy.calculate.treecompare.weighted_robinson_foulds_distance(dendro_tree, dendro_tree_dodo)

    assert np.isclose(diff, 0.0)
