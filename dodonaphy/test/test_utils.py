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
        dir2, torch.tensor([1.]))
    assert 1.438266 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary():
    r1 = torch.tensor([0.])
    r2 = torch.tensor([.9999999])
    dir1 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dir2 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dist = utils.hyperbolic_distance(
        r1, r2, dir1,
        dir2, torch.tensor([1.]))
    assert 16.635532 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_zero():
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dist = utils.hyperbolic_distance(
        torch.tensor([0.5]), torch.tensor([0.5]), dir1,
        dir2, torch.tensor([1.]))
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
