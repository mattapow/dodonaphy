#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:55:22 2021

@author: 151569
"""
import src.hyperboloid as hyp
import torch
from pytest import approx


def test_poincare_to_hyper():
    loc_poin = torch.tensor([[0.5, 0.3], [-0.1, 0.7]])
    loc_hyp = hyp.poincare_to_hyper(loc_poin)

    assert hyp.lorentz_product(loc_hyp[0, :]).item() == approx(-1)
    assert hyp.lorentz_product(loc_hyp[1, :]).item() == approx(-1)


def test_tangent_to_hyper():
    dim = 2
    mu = torch.tensor([3, 2, 2])
    v_tilde = torch.tensor([2.4, -0.3])
    z = hyp.tangent_to_hyper(mu, v_tilde, dim)
    assert torch.isclose(hyp.lorentz_product(z), torch.as_tensor(-1).double())


def test_p2t02p():
    input = torch.tensor([[0.1, 0.3, 0.4]]).double()
    output = hyp.t02p(hyp.p2t0(input))
    assert approx(input.data, output.data)


def test_t02p2t0():
    input = torch.tensor([[10.0, 0.3, -0.44]]).double()
    output = hyp.p2t0(hyp.t02p(input))
    assert approx(input.data, output.data)


def test_jacobian_default_mu():
    x_t0 = torch.tensor([[1.4, -0.3, -0.8]]).double()
    x_poin, jacobian0 = hyp.t02p(x_t0, get_jacobian=True)
    x_t0_2, jacobian2 = hyp.p2t0(x_poin, get_jacobian=True)
    assert torch.allclose(x_t0, x_t0_2)
    assert torch.allclose(jacobian0, -jacobian2)


def test_jacobian_mu_0():
    x_t0 = torch.tensor([[-0.4, -0.0, -1.0]]).double()
    mu = torch.zeros_like(x_t0).double()
    x_poin, jacobian0 = hyp.t02p(x_t0, mu, get_jacobian=True)
    mu = torch.hstack((torch.ones(1, 1), mu))
    x_t0_2, jacobian2 = hyp.p2t0(x_poin, mu, get_jacobian=True)
    assert torch.allclose(x_t0, x_t0_2)
    assert torch.allclose(jacobian0, -jacobian2)


def test_jacobian_mu_1():
    x_t0 = torch.tensor([[-6.4, 0.0, 1.0]]).double()
    mu = torch.tensor([[0.2, 0.3, -0.5]]).double()
    x_poin, jacobian_0 = hyp.t02p(x_t0, mu, get_jacobian=True)
    mu_2 = hyp.up_to_hyper(mu)
    x_t0_2, jacobian_2 = hyp.p2t0(x_poin, mu_2, get_jacobian=True)
    assert torch.allclose(x_t0, x_t0_2, atol=0.00001)
    assert torch.allclose(jacobian_0, -jacobian_2)
