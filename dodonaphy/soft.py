import torch

eps = torch.finfo(torch.double).eps
eps_torch = torch.tensor(eps)


def unravel_index(index, shape):
    row = torch.div(index, shape[-1], rounding_mode='trunc')
    col = torch.fmod(index, shape[-1])
    return row, col


def sort(s, tau):
    s_sorted = s.sort(descending=True, dim=1)[0]
    pairwise_distances = (s.transpose(1, 2) - s_sorted).abs().neg() / tau
    P_hat = pairwise_distances.softmax(-1)
    return P_hat


def min(s, tau):
    assert s.ndim == 1
    s_3d = s.unsqueeze(0).unsqueeze(-1)
    P_hat = sort(s_3d, tau)
    return (P_hat @ s)[0, -1]


def max(s, tau):
    return -min(-s, tau)


def clamp_pos(s, tau, epsilon=eps_torch):
    assert s.ndim == 0
    s = s.unsqueeze(-1)
    s = torch.cat((s, epsilon.unsqueeze(-1)))
    return max(s, tau)


def argmin(Q, tau):
    "Soft argmin function of matrix that breaks ties with cumsum."
    assert Q.ndim == 2
    # there may be ties in Q
    Q_flat_ties = Q.view(-1)
    P_hat_ties = sort(Q_flat_ties.unsqueeze(0).unsqueeze(-1), tau)
    # choose the last of any ties by multiplying by the cumulative sum
    Q_flat = P_hat_ties[:, -1] * torch.cumsum(P_hat_ties[:, -1], -1)
    P_hat = sort(-Q_flat.unsqueeze(-1), tau)

    # permutation matrix to index
    unravel_indices = torch.arange(Q.numel(), dtype=Q.dtype)
    soft_indices = (P_hat[:, -1] * unravel_indices).sum((-1, 0))
    # flattened index to (row, col)
    soft_row, soft_col = unravel_index(soft_indices, Q.shape)
    return soft_row, soft_col


def index_to_hard_one_hot(index: torch.double, size):  
    """ Convert a floating point of an index to a one hot vector.
    No gradients passed."""
    one_hot = torch.zeros(size, dtype=torch.double)
    one_hot.scatter_(0, index.round().unsqueeze(0).long(), 1.0)
    return one_hot
