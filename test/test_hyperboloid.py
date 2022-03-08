#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:55:22 2021

@author: 151569
"""
import torch
from dodonaphy import utils, Chyp_np, Chyp_torch
from pytest import approx
import numpy as np


# def test_hyperbolic_distance():
#     r1 = torch.tensor([0.3])
#     r2 = torch.tensor([0.6])
#     dir1 = torch.tensor(
#         [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
#     )
#     dir2 = torch.tensor([torch.as_tensor(-0.5), torch.as_tensor(np.sqrt(0.75))])
#     x1 = utils.dir_to_cart(r1, dir1)
#     x2 = utils.dir_to_cart(r2, dir2)
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     dist2 = Chyp_torch.hyperbolic_distance_lorentz(x1.squeeze(), x2.squeeze(), -torch.tensor([1.0]))
#     assert dist2.item() == approx(dist.item(), 0.0001)


# def test_hyperbolic_distance_boundary1():
#     r1 = torch.tensor([0.99999997])
#     r2 = torch.tensor([0.99999997])
#     dir1 = torch.tensor(
#         [torch.as_tensor(-1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))],
#         dtype=torch.double,
#     )
#     dir2 = torch.tensor(
#         [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
#     )
#     x1 = utils.dir_to_cart(r1, dir1)
#     x2 = utils.dir_to_cart(r2, dir2)
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     dist2 = Chyp_torch.hyperbolic_distance_lorentz(x1.squeeze(), x2.squeeze(), -torch.tensor([1.0]))
#     assert dist2.item() == approx(dist.item(), 0.0001)


# def test_hyperbolic_distance_boundary2():
#     r1 = torch.tensor([0.99999997])
#     r2 = torch.tensor([0.99999997])
#     dir1 = torch.tensor(
#         [torch.as_tensor(-0.0), torch.as_tensor(-1.0)], dtype=torch.double
#     )
#     dir2 = torch.tensor(
#         [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
#     )
#     x1 = utils.dir_to_cart(r1, dir1)
#     x2 = utils.dir_to_cart(r2, dir2)
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     dist2 = Chyp_torch.hyperbolic_distance_lorentz(x1.squeeze(), x2.squeeze(), -torch.tensor([1.0]))
#     assert dist2.item() == approx(dist.item(), 0.0001)


# def test_hyperbolic_distance_boundary3():
#     r1 = torch.tensor([0.99999999])
#     r2 = torch.tensor([0.99999999])
#     dir1 = torch.tensor(
#         [torch.as_tensor(-1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))],
#         dtype=torch.double,
#     )
#     dir2 = torch.tensor(
#         [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
#     )
#     x1 = utils.dir_to_cart(r1, dir1)
#     x2 = utils.dir_to_cart(r2, dir2)
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     dist2 = Chyp_torch.hyperbolic_distance_lorentz(x1.squeeze(), x2.squeeze(), -torch.tensor([1.0]))
#     assert dist2.item() == approx(dist.item(), 0.0001)


# def test_hyperbolic_distance_boundary_close():
#     r1 = torch.tensor([0.9999999999])
#     r2 = torch.tensor([0.99999999991])
#     dir1 = torch.tensor(
#         [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
#     )
#     dir2 = torch.tensor(
#         [torch.as_tensor(0.0), torch.as_tensor(1.0)], dtype=torch.double
#     )
#     x1 = utils.dir_to_cart(r1, dir1)
#     x2 = utils.dir_to_cart(r2, dir2)
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     dist2 = Chyp_torch.hyperbolic_distance_lorentz(x1.squeeze(), x2.squeeze(), -torch.tensor([1.0]))
#     assert dist2.item() == approx(dist.item(), 0.0001)


# def test_hyperbolic_distance_zero():
#     x1 = torch.tensor(
#         [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
#     ) / 2.
#     x2 = torch.tensor(
#         [torch.as_tensor(1.0 / np.sqrt(2)), torch.as_tensor(1.0 / np.sqrt(2))]
#     ) / 2.
#     dist = Chyp_torch.hyperbolic_distance(x1, x2, -torch.tensor([1.0]))
#     assert 0.0 == approx(dist.item(), abs=0.05)

#     x_t0 = np.array([[0.2, .3, .4], [-.4, 5, .3], [0., 6., -3]])
#     x_hyp = np.zeros((3, 4))
#     x_poin = np.zeros((3, 3))
#     for i in range(3):
#         x_hyp[i, :] = Chyp_np.project_up(x_t0[i, :])
#         x_poin[i, :] = Chyp_np.hyper_to_poincare(x_hyp[i, :])
#     pdm = Chyp_np.get_pdm(x_hyp)
#     pdm_2 = Chyp_torch.get_pdm(torch.from_numpy(x_poin))
#     assert np.allclose(pdm, pdm_2.detach().numpy(), atol=1e-6)


def test_poincare_to_hyper():
    loc_poin = torch.tensor([[0.5, 0.3], [-0.1, 0.7]])
    loc_hyp = Chyp_torch.poincare_to_hyper(loc_poin)

    assert Chyp_torch.lorentz_product(loc_hyp[0, :]).item() == approx(-1)
    assert Chyp_torch.lorentz_product(loc_hyp[1, :]).item() == approx(-1)


def test_tangent_to_hyper():
    dim = 2
    mu = torch.tensor([3, 2, 2], dtype=torch.double)
    v_tilde = torch.tensor([2.4, -0.3], dtype=torch.double)
    z = Chyp_torch.tangent_to_hyper(mu, v_tilde, dim)
    assert torch.isclose(Chyp_torch.lorentz_product(z), torch.as_tensor(-1).double())


# def test_p2t02p():
#     input_data = torch.tensor([[0.1, 0.3, 0.4]]).double()
#     output = Chyp_torch.t02p(Chyp_torch.p2t0(input_data))
#     assert approx(input_data.data, output.data)


# def test_t02p2t0():
#     input_data = torch.tensor([[10.0, 0.3, -0.44]]).double()
#     output = Chyp_torch.p2t0(Chyp_torch.t02p(input_data))
#     assert approx(input_data.data, output.data)


def test_jacobian_default_mu():
    x_t0 = torch.tensor([[1.4, -0.3, -0.8]]).double()
    x_poin, jacobian0 = Chyp_torch.t02p(x_t0, get_jacobian=True)
    x_t0_2, jacobian2 = Chyp_torch.p2t0(x_poin, get_jacobian=True)
    assert torch.allclose(x_t0, x_t0_2)
    assert torch.allclose(jacobian0, -jacobian2)
