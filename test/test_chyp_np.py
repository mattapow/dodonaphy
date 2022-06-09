import numpy as np
import torch
from dodonaphy import Chyp_np


def test_dist_gradient():
    x_np = np.array([
        [0.1, 0.2],
        [-0.1, 0.1],
        [0.06, -0.2]
        ])
    n, dim = x_np.shape
    k_np = -np.ones(1)

    _, jacobian = Chyp_np.get_pdm(x_np, curvature=k_np, get_jacobian=True)

    x_torch = torch.from_numpy(x_np)
    k_torch = torch.from_numpy(k_np)
    x_sheet = project_up_2d(x_torch)

    def torch_distance_flat(x_flat):
        x_sheet = x_flat.view((n, dim+1))
        return torch_distance(x_sheet)

    def torch_distance(x_sheet):
        """Copy of Chyp_np.get_pdm for pytorch."""
        X = x_sheet @ x_sheet.T
        u_tilde = torch.sqrt(torch.diagonal(X) + 1)
        H = X - torch.outer(u_tilde, u_tilde)
        H = torch.clamp(H, max=-(1 + 1e-8))
        D = 1.0 / torch.sqrt(-k_torch) * torch.acosh(-H)
        # flatten without upper trianglular elements
        return D[np.triu_indices(len(D), k=1)]
    jacobian_torch = torch.autograd.functional.jacobian(
        torch_distance_flat, (x_sheet.flatten())
    )
    print(f"Analytical: {torch.tensor(jacobian)}")
    print(f"Pytorch: {jacobian_torch}")
    print(f"Analytical: {jacobian.shape}")
    print(f"Pytorch: {jacobian_torch.shape}")
    assert np.allclose(jacobian, jacobian_torch)

def project_up_2d(loc):
    z = torch.unsqueeze(torch.sqrt(torch.sum(torch.pow(loc, 2), 1) + 1), 1)
    return torch.cat((z, loc), dim=1)

def test_up_jacobian():
    loc = np.array([
        [0.1, -.3],
        [0.8, 1],
        [-.04, 10],
        [-0.07, 0.09]
    ])
    loc_torch = torch.from_numpy(loc)

    jacobian_analytic = Chyp_np.project_up_jacobian(loc)
    jacobian_torch = torch.autograd.functional.jacobian(
        project_up_pytorch, (loc_torch.flatten())
    )
    assert np.allclose(jacobian_analytic, jacobian_torch)

def project_up_pytorch(loc_flat):
    """A copy in pytorch.
    Warning: Only for test_up_jacobian()
    Hard coded shape for this test only n_taxa=4, n_dim=2.
    """
    loc = loc_flat.view((4, 2))
    z = torch.sqrt(torch.sum(torch.pow(loc, 2), dim=-1, keepdim=True) + 1)
    return torch.cat((z, loc), dim=-1).flatten()
