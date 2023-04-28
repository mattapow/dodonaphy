import torch
from dodonaphy import soft


def test_soft_min():
    s = torch.tensor([5, 2, 4.4, 0.5, 10], requires_grad=True)
    s_min = soft.min(s, 1e-3)
    expected_min = torch.tensor([0.5])
    assert torch.isclose(s_min, expected_min)
    assert s.requires_grad


def test_clamp_pos():
    s = torch.tensor(-3.0, requires_grad=True)
    epsilon = torch.tensor(0.0001)
    s_pos = soft.clamp_pos(s, 1e-3, epsilon=epsilon)
    assert s_pos.requires_grad
    assert torch.isclose(s_pos, epsilon)


def test_soft_sort():
    s = (
        torch.tensor([3, 2, 1], dtype=torch.double, requires_grad=True)
        .unsqueeze(dim=0)
        .unsqueeze(dim=-1)
    )
    tau = 0.5
    expected = torch.tensor([2.8509, 2.0000, 1.1491], dtype=torch.double)
    permute = soft.sort(s, tau)
    calculated = permute @ s
    assert torch.allclose(calculated.squeeze(), expected, rtol=1e-3)
    assert calculated.requires_grad


def test_soft_sort_1d():
    input_arr = torch.tensor([2, 5.5, 3, -1, 0], requires_grad=True)
    permute = soft.sort(input_arr.unsqueeze(-1).unsqueeze(0), tau=0.000001).squeeze()
    output = permute @ input_arr
    correct = torch.tensor([5.5, 3, 2, 0, -1])
    assert torch.allclose(output, correct)
    assert output.requires_grad


def test_soft_sort_2d():
    input_arr = torch.tensor(
        [[0, 50, 38, 34], [50, 0, 38, 34], [38, 38, 0, 40], [34, 34, 40, 0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    permute = soft.sort(input_arr.unsqueeze(-1), tau=0.000001)

    row0_argmax = permute[0, 0]
    assert torch.allclose(row0_argmax[1], torch.ones(1))
    assert torch.allclose(sum(row0_argmax), torch.ones(1))
    assert row0_argmax.requires_grad
    row1_argmax = permute[1, 0]
    assert torch.allclose(row1_argmax[0], torch.ones(1))
    assert torch.allclose(sum(row1_argmax), torch.ones(1))
    assert row1_argmax.requires_grad
    row2_argmax = permute[2, 0]
    assert torch.allclose(row2_argmax[3], torch.ones(1))
    assert torch.allclose(sum(row2_argmax), torch.ones(1))
    assert row2_argmax.requires_grad
