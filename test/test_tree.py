from dendropy import Tree
import dendropy
import torch
import numpy as np

from dodonaphy import tree as treeFunc


def test_dendropy_to_pb():
    data = "(6:6.000000e-02,((5:5.000000e-02,2:2.000000e-02):17.000000e-02,\
        (4:4.000000e-02,3:3.000000e-02):18.000000e-02):19.000000e-02,1:1.000000e-02);"
    dendo_tree = Tree.get(data=data, schema="newick")
    peel, blens = treeFunc.dendrophy_to_pb(dendo_tree, offset=1)

    true_peel = np.array([[ 4,  1,  6],
       [ 3,  2,  7],
       [ 6,  7,  8],
       [ 5,  8,  9],
       [ 0,  9, 10]])
    assert np.allclose(peel, true_peel)

    true_blens = torch.tensor([0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600, 0.1700, 0.1800, 0.1900,
        0.0000], dtype=torch.float64)
    assert torch.allclose(blens, true_blens)
