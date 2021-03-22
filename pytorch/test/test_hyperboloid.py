#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 07:55:22 2021

@author: 151569
"""
import torch
import dodonaphy.hyperboloid as hyp
from pytest import approx


def test_sample_dimensions_3D():
    dim = 3
    mu = torch.ones(dim+1)
    cov = torch.ones(dim)

    sample = hyp.sample_normal_hyper(mu, cov, dim)
    assert sample.shape[0] == approx(dim+1)


def test_sample_dimensions_7D():
    dim = 7
    mu = torch.ones(dim+1)
    cov = torch.ones(dim)

    sample = hyp.sample_normal_hyper(mu, cov, dim)
    assert sample.shape[0] == approx(dim+1)


def test_embed_star_hyperboloid_valid_out():
    n_seqs = 6
    z = 5
    X = hyp.embed_star_hyperboloid_2d(z, n_seqs)

    for i in range(n_seqs):
        assert torch.isclose(hyp.lorentz_product(X[i, :]), -torch.ones(1))
