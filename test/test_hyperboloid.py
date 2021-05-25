#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:55:22 2021

@author: 151569
"""
import torch
import src.hyperboloid as hyp
from pytest import approx


def test_embed_star_hyperboloid_valid_out():
    n_seqs = 6
    z = 5
    X = hyp.embed_star_hyperboloid_2d(z, n_seqs)

    for i in range(n_seqs):
        assert torch.isclose(hyp.lorentz_product(X[i, :]), -torch.ones(1))


def test_poincare_to_hyper():
    loc_poin = torch.tensor([[.5, .3],
                             [-.1, .7]])
    loc_hyp = hyp.poincare_to_hyper(loc_poin)

    assert hyp.lorentz_product(loc_hyp[0, :]).item() == approx(-1)
    assert hyp.lorentz_product(loc_hyp[1, :]).item() == approx(-1)


def test_tangent_to_hyper():
    dim = 2
    mu = torch.tensor([3, 2, 2])
    v_tilde = torch.tensor([2.4, -.3])
    z = hyp.tangent_to_hyper(mu, v_tilde, dim)
    assert torch.isclose(hyp.lorentz_product(z), torch.as_tensor(-1).double())
