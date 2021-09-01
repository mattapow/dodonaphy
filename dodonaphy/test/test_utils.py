import pytest
import torch
import numpy as np
from pytest import approx
from src import utils


def test_hyperbolic_distance():
    r1 = torch.tensor([0.3])
    r2 = torch.tensor([.6])
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(-.5), torch.as_tensor(np.sqrt(0.75))])
    dist = utils.hyperbolic_distance(
        r1, r2, dir1,
        dir2, -torch.tensor([1.]))
    assert 1.438266 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary():
    r1 = torch.tensor([0.])
    r2 = torch.tensor([.9999999])
    dir1 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dir2 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dist = utils.hyperbolic_distance(
        r1, r2, dir1,
        dir2, -torch.tensor([1.]))
    assert 16.635532 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_zero():
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dist = utils.hyperbolic_distance(
        torch.tensor([0.5]), torch.tensor([0.5]), dir1,
        dir2, -torch.tensor([1.]))
    assert 0. == pytest.approx(dist.item(), abs=0.05)


def test_dir_to_cart_1d():
    u = torch.tensor(10.)
    r = torch.norm(u)
    directional = u / r
    loc = utils.dir_to_cart(r, directional)
    assert loc == approx(u)


def test_dir_to_cart_5d():
    u = torch.tensor((10., 2.3, 43., -4., 4.5))
    r = torch.norm(u)
    directional = u / r
    loc = utils.dir_to_cart(r, directional)
    assert loc == approx(u)


def test_euclidean_distance():
    r1 = torch.tensor([0.3])
    r2 = torch.tensor([.6])
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(-.5), torch.as_tensor(np.sqrt(0.75))])
    dist = utils.hyperbolic_distance(
        r1, r2, dir1,
        dir2, torch.zeros(1))
    x1 = r1*dir1
    x2 = r2*dir2
    norm = torch.sum(x2**2-x1**2)**.5
    assert norm == pytest.approx(dist.item(), 0.0001)


def test_pdm_euclidean():
    n_tips = 4
    leaf_r = torch.from_numpy(np.random.uniform(0, 1, n_tips))
    leaf_dir = torch.from_numpy(np.random.normal(0, 1, (n_tips, 2)))
    leaf_norm = torch.sum(leaf_dir, axis=1, keepdims=True)
    leaf_dir /= leaf_norm
    _ = utils.get_pdm_tips(leaf_r, leaf_dir, curvature=torch.zeros(1))
