import dendropy
import torch
from pytest import approx
from src import tree as treeFunc
from src.base_model import BaseModel


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
