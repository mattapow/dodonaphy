import numpy as np
import torch
from dodonaphy import Chyp_np


def test_dist_gradient():
    x_np = np.array([[0.1, 0.2, -0.04], [-0.1, -0.04, 0.5]])
    k_np = -np.ones(1)

    _, jacobian = Chyp_np.get_pdm(x_np, curvature=k_np, get_jacobian=True)

    x_torch = torch.from_numpy(x_np)
    k_torch = torch.from_numpy(k_np)
    jacobian_torch = torch.autograd.functional.jacobian(
        torch_distance, (x_torch, k_torch)
    )
    assert np.allclose(jacobian, jacobian_torch[0][1][0])


def torch_distance(x, curvature):
    """Copy of Chyp_np.get_pdm for pytorch."""
    x_sheet = project_up_2d(x)
    X = x_sheet @ x_sheet.T
    u_tilde = torch.sqrt(torch.diagonal(X) + 1)
    H = X - torch.outer(u_tilde, u_tilde)
    H = torch.clamp(H, max=-(1 + 1e-16))
    D = 1.0 / torch.sqrt(-curvature) * torch.acosh(-H)
    return D


def project_up_2d(loc):
    z = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(loc, 2), 1) + 1), 1)
    return torch.cat((z, loc), dim=1)
