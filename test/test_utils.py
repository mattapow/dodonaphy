import numpy as np
import pytest
import torch
from pytest import approx
from src import utils, Cutils


def test_hyperbolic_distance():
    r1 = torch.tensor([0.3])
    r2 = torch.tensor([0.6])
    dir1 = torch.tensor(
        [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
    )
    dir2 = torch.tensor([torch.as_tensor(-0.5), torch.as_tensor(np.sqrt(0.75))])
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, -torch.tensor([1.0]))
    x1 = utils.dir_to_cart(r1, dir1)
    x2 = utils.dir_to_cart(r2, dir2)
    dist2 = Cutils.hyperbolic_distance_lorentz(x1, x2, -torch.tensor([1.0]))
    assert dist2.item() == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary1():
    r1 = torch.tensor([0.99999997])
    r2 = torch.tensor([0.99999997])
    dir1 = torch.tensor(
        [torch.as_tensor(-1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))],
        dtype=torch.double,
    )
    dir2 = torch.tensor(
        [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
    )
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, -torch.tensor([1.0]))
    x1 = utils.dir_to_cart(r1, dir1)
    x2 = utils.dir_to_cart(r2, dir2)
    dist2 = Cutils.hyperbolic_distance_lorentz(x1, x2, -torch.tensor([1.0]))
    assert dist2.item() == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary2():
    r1 = torch.tensor([0.99999997])
    r2 = torch.tensor([0.99999997])
    dir1 = torch.tensor(
        [torch.as_tensor(-0.0), torch.as_tensor(-1.0)], dtype=torch.double
    )
    dir2 = torch.tensor(
        [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
    )
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, -torch.tensor([1.0]))
    x1 = utils.dir_to_cart(r1, dir1)
    x2 = utils.dir_to_cart(r2, dir2)
    dist2 = Cutils.hyperbolic_distance_lorentz(x1, x2, -torch.tensor([1.0]))
    assert dist2.item() == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary3():
    r1 = torch.tensor([0.99999999])
    r2 = torch.tensor([0.99999999])
    dir1 = torch.tensor(
        [torch.as_tensor(-1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))],
        dtype=torch.double,
    )
    dir2 = torch.tensor(
        [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
    )
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, -torch.tensor([1.0]))
    x1 = utils.dir_to_cart(r1, dir1)
    x2 = utils.dir_to_cart(r2, dir2)
    dist2 = Cutils.hyperbolic_distance_lorentz(x1, x2, -torch.tensor([1.0]))
    assert dist2.item() == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary_close():
    r1 = torch.tensor([0.9999999999])
    r2 = torch.tensor([0.99999999991])
    dir1 = torch.tensor(
        [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
    )
    dir2 = torch.tensor(
        [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
    )
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, -torch.tensor([1.0]))
    x1 = utils.dir_to_cart(r1, dir1)
    x2 = utils.dir_to_cart(r2, dir2)
    dist2 = Cutils.hyperbolic_distance_lorentz(x1, x2, -torch.tensor([1.0]))
    assert dist2.item() == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_zero():
    dir1 = torch.tensor(
        [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
    )
    dir2 = torch.tensor(
        [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
    )
    dist = Cutils.hyperbolic_distance(
        torch.tensor([0.5]), torch.tensor([0.5]), dir1, dir2, -torch.tensor([1.0])
    )
    assert 0.0 == pytest.approx(dist.item(), abs=0.05)


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
    dist = Cutils.hyperbolic_distance(r1, r2, dir1, dir2, torch.zeros(1))
    x1 = r1 * dir1
    x2 = r2 * dir2
    norm = torch.norm(x2 - x1)
    assert norm == pytest.approx(dist.item(), 0.0001)


def test_pdm_euclidean():
    n_tips = 4
    leaf_r = torch.from_numpy(np.random.uniform(0, 1, n_tips))
    leaf_dir = torch.from_numpy(np.random.normal(0, 1, (n_tips, 2)))
    leaf_norm = torch.tile(torch.norm(leaf_dir, dim=1), (2, 1)).T
    leaf_dir /= leaf_norm
    _ = utils.get_pdm_tips(leaf_r, leaf_dir, curvature=-torch.zeros(1))
