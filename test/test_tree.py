from dendropy import Tree
import dendropy
import torch

from dodonaphy import tree as treeFunc


def test_dendropy_to_pb():
    data = "(6:2.000000e-02,((5:2.000000e-02,2:2.000000e-02):2.000000e-02,\
        (4:2.000000e-02,3:2.000000e-02):2.000000e-02):2.000000e-02,1:2.000000e-02);"
    dendo_tree = Tree.get(data=data, schema="newick")
    peel, blens = treeFunc.dendrophy_to_pb(dendo_tree)

    for i in range(4):
        assert peel[i, 2] >= 5
    assert sum(sum(peel)) == 54

    assert torch.allclose(blens[:-1], 0.02*torch.ones((9,), dtype=torch.double))
    assert blens[-1] == 0

test_dendropy_to_pb()