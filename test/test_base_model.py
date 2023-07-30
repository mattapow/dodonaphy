import torch
from dodonaphy.base_model import BaseModel


def test_compute_prior_gamma_dir_small():
    blens = torch.full([3], 0.1, requires_grad=True)
    prior = BaseModel.compute_prior_gamma_dir(blens)
    assert prior.requires_grad


def test_compute_prior_gamma_dir_big():
    blens = torch.full([100], 0.1, requires_grad=True)
    prior = BaseModel.compute_prior_gamma_dir(blens)
    assert prior.requires_grad

def test_compute_prior_exponential():
    blens = torch.full([100], 0.1, requires_grad=True)
    prior = BaseModel.compute_prior_exponential(blens)
    assert prior.requires_grad
    assert prior.size() == torch.Size([])
