import pytest
import torch

from dodonaphy.utils import utilFunc


def test_hyperbolic_distance():
    dist = utilFunc.hyperbolic_distance(torch.tensor([0.5]), torch.tensor([0.6]), torch.tensor([0.1, 0.3]),
                                        torch.tensor([0.5, 0.5]),
                                        torch.tensor([1.]))
    assert 1.777365 == pytest.approx(dist.item(), 0.0001)
