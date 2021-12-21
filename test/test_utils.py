import numpy as np
import pytest
import torch
from pytest import approx
from dodonaphy import utils, Chyperboloid, Chyperboloid_np

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
    dist = Chyperboloid.hyperbolic_distance(r1, r2, dir1, dir2, torch.zeros(1))
    x1 = r1 * dir1
    x2 = r2 * dir2
    norm = torch.norm(x2 - x1)
    assert norm == pytest.approx(dist.item(), 0.0001)


def test_pdm_euclidean():
    n_tips = 4
    leaf_r = np.random.uniform(0, 1, n_tips)
    leaf_dir = np.random.normal(0, 1, (n_tips, 2))
    leaf_norm = np.tile(np.linalg.norm(leaf_dir, axis=1), (2, 1)).T
    leaf_dir /= leaf_norm
    _ = Chyperboloid_np.get_pdm(leaf_r, leaf_dir, curvature=0.0, dtype="numpy")
