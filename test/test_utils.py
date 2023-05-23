import numpy as np
import torch
from pytest import approx
from dodonaphy import utils, Cutils, Chyp_torch, Chyp_np
import random
from dendropy.simulate import treesim


def test_dir_to_cart_1d():
    u = torch.tensor(10.0)
    r = torch.norm(u)
    directional = u / r
    loc = utils.dir_to_cart(r, directional)
    assert loc == approx(u)


def test_dir_to_cart_5d():
    u = torch.tensor((10.0, 2.3, 43.0, -4.0, 4.5))
    r = torch.norm(u)
    directional = u / r
    loc = utils.dir_to_cart(r, directional)
    assert loc == approx(u)


def test_euclidean_distance():
    r1 = torch.tensor([0.3])
    r2 = torch.tensor([0.6])
    dir1 = torch.tensor(
        [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))],
        dtype=torch.double,
    )
    dir2 = torch.tensor(
        [torch.as_tensor(-0.5), torch.as_tensor(np.sqrt(0.75))], dtype=torch.double
    )
    x1 = r1 * dir1
    x2 = r2 * dir2
    dist = Chyp_torch.hyperbolic_distance(x1, x2, torch.zeros(1))
    norm = torch.norm(x2 - x1)
    assert norm == approx(dist.item(), 0.0001)


def test_pdm_almost_euclidean():
    n_tips = 4
    leaf_r = np.random.uniform(0, 1, n_tips)
    leaf_dir = np.random.normal(0, 1, (n_tips, 2))
    leaf_norm = np.tile(np.linalg.norm(leaf_dir, axis=1), (2, 1)).T
    leaf_dir /= leaf_norm
    leaf_x = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    curvature = -0.01
    _ = Chyp_np.get_pdm(leaf_x, curvature=curvature)


def test_LogDirPrior():
    blen = torch.full([5], 0.1, requires_grad=True)
    aT = torch.ones(1)
    bT = torch.full((1,), 0.1)
    a = torch.ones(1)
    c = torch.ones(1)
    prior = utils.LogDirPrior(blen, aT, bT, a, c)
    assert prior.requires_grad


def test_tip_distances():
    n_taxa = 6
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=2.0,
        death_rate=0.5,
        num_extant_tips=n_taxa,
        rng=rng,
    )
    dists = utils.tip_distances(simtree)
    assert dists.shape == (6, 6)


def test_all_distances():
    n_taxa = 6
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=2.0,
        death_rate=0.5,
        num_extant_tips=n_taxa,
        rng=rng,
    )
    dists = utils.all_distances(simtree)
    assert dists.shape == (11, 11)
