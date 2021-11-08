import numpy as np
import pytest
import torch
from dodonaphy import peeler, poincare, utils, Cutils


def test_make_peel_simple():
    # Connect three evenly spaced leaves
    # It seems this is only coming about when S=3
    # Issue coming from utils.py#L371
    S = 3
    leaf_r = 0.5 * torch.ones(S)
    leaf_theta = torch.tensor([np.pi / 6, 0.0, -np.pi / 9])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    # internal node with angle in between nodes 0 and 1
    int_r = torch.tensor([0.25])
    int_theta = torch.tensor([np.pi / 12])
    int_dir = utils.angle_to_directional(int_theta)

    # Connect nodes
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    # See utilFunc.plot_tree:
    # Tree should connect 0 and 1 to internal node 3
    # root node 4, should connect to 0 and 3.
    assert np.allclose(peel, np.array([[1, 2, 3], [0, 3, 4]]))


def test_make_peel_dogbone():
    # Take 4 leaves and form a dogbone tree
    leaf_r = torch.tensor([0.5, 0.5, 0.8, 0.8])
    leaf_theta = torch.tensor([np.pi / 6, 0.0, -np.pi * 0.7, -np.pi * 0.8])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    int_r = torch.tensor([0.25, 0.4])
    int_theta = torch.tensor([np.pi / 12, -np.pi * 0.75])
    int_dir = utils.angle_to_directional(int_theta)

    # make a tree
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    assert np.allclose(peel, np.array([[2, 3, 5], [1, 5, 4], [0, 4, 6]]))


def test_make_peel_first_leaf_connection():
    leaf_r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    leaf_theta = torch.tensor([0.0, np.pi / 6, -np.pi / 6, np.pi])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    int_r = torch.tensor([0.3, 0])
    int_theta = torch.tensor([np.pi / 6, 0])
    int_dir = utils.angle_to_directional(int_theta)

    # make a tree
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    correct1 = np.allclose(peel, np.array([[3, 2, 5], [1, 5, 4], [0, 4, 6]]))
    correct2 = np.allclose(peel, np.array([[2, 3, 5], [1, 5, 4], [0, 4, 6]]))

    assert correct1 or correct2


def test_make_peel_example1():
    leaf_locs = torch.tensor(
        [
            [5.5330e-02, 4.0385e-02],
            [6.0270e-02, 4.4329e-02],
            [-1.1253e-01, -1.5676e-01],
            [1.0916e-01, -7.2296e-02],
            [5.9408e-02, 4.0677e-02],
            [-4.0814e-02, -3.1838e-01],
        ]
    )

    int_locs = torch.tensor(
        [[0.0352, 0.0405], [0.0437, 0.0144], [0.0375, -0.070], [-0.0633, -0.1595]],
        dtype=torch.float64,
    )

    leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
    int_r, int_dir = utils.cart_to_dir(int_locs)
    _ = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)


def test_make_peel_incentre():
    leaf_locs = torch.tensor(
        [[0.06644704, 0.18495312, -0.03839615], [-0.12724270, -0.02395994, 0.20075329]]
    )
    lca = poincare.hyp_lca(leaf_locs[0], leaf_locs[1])
    d0_1 = Cutils.hyperbolic_distance_lorentz(leaf_locs[0], leaf_locs[1])
    d0_lca = Cutils.hyperbolic_distance_lorentz(leaf_locs[0], lca)
    d1_lca = Cutils.hyperbolic_distance_lorentz(leaf_locs[1], lca)
    d0_lca_d1 = d0_lca + d1_lca
    assert pytest.approx(d0_1, d0_lca_d1)


def test_make_peel_geodesic_dogbone():
    leaf_r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = utils.angle_to_directional(leaf_theta)
    leaf_locs = utils.dir_to_cart(leaf_r, leaf_dir)
    peel, int_locs = peeler.make_peel_incentre(leaf_locs)
    expected_peel = np.array([[1, 0, 4], [3, 2, 5], [4, 5, 6]])

    assert np.allclose(peel, expected_peel)


def test_make_peel_geodesic_example0():
    leaf_locs = torch.tensor(
        [
            (0.05909264, 0.16842421, -0.03628194),
            (0.08532969, -0.07187002, 0.17884444),
            (-0.11422830, 0.01955054, 0.14127290),
            (-0.06550432, 0.07029946, -0.14566249),
            (-0.07060744, -0.12278600, -0.17569585),
            (0.11386343, -0.03121063, -0.18112418),
        ]
    )
    peel, int_locs = peeler.make_peel_incentre(leaf_locs)


def test_make_peel_geodesic_example1():
    leaf_locs = torch.tensor(
        [
            [1.5330e-02, 1.0385e-02],
            [-6.0270e-02, 4.4329e-02],
            [-1.1253e-01, -1.5676e-01],
            [1.0916e-01, -7.2296e-02],
            [5.9408e-02, 3.0677e-04],
            [-4.0814e-02, -3.1838e-01],
        ]
    )
    peel, int_locs = peeler.make_peel_incentre(leaf_locs)
    for i in range(5):
        assert int(peel[i][0]) is not int(peel[i][1])
        assert int(peel[i][0]) is not int(peel[i][2])
        assert int(peel[i][1]) is not int(peel[i][2])


def test_make_peel_geodesic_example2():
    leaf_locs = torch.tensor(
        [
            [2.09999997e-02, -2.09410252e-01],
            [1.01784302e-01, 3.18292447e-02],
            [-9.41199092e-02, 7.48080376e-02],
            [-1.85000000e-02, -1.25415666e-01],
            [-3.39999999e-02, -1.52410744e-01],
            [-1.40397397e-02, 1.47278753e-01],
        ]
    )
    peel, int_locs = peeler.make_peel_incentre(leaf_locs)
    for i in range(5):
        assert int(peel[i][0]) is not int(peel[i][1])
        assert int(peel[i][0]) is not int(peel[i][2])
        assert int(peel[i][1]) is not int(peel[i][2])


def test_nj():
    leaf_r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    pdm = Cutils.get_pdm_torch(leaf_r, leaf_dir)
    peel, blens = peeler.nj(pdm)

    peel_check = []
    peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [1, 4, 5], [0, 5, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 2, 5], [5, 3, 6]]))
    assert sum(
        peel_check
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."
    assert torch.isclose(sum(blens).float(), torch.tensor(2.0318).float(), atol=0.001)


def test_nj_uneven():
    leaf_r = torch.tensor([0.1, 0.2, 0.3, 0.4])
    leaf_theta = torch.tensor([np.pi / 2, -np.pi / 10, np.pi, -np.pi * 6 / 8])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    pdm = Cutils.get_pdm_torch(leaf_r, leaf_dir)
    peel, blens = peeler.nj(pdm)
    peel_check = []
    peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [1, 4, 5], [0, 5, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 2, 5], [5, 3, 6]]))
    assert sum(
        peel_check
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."


def test_compute_Q():
    pdm = torch.zeros((5, 5)).double()
    pdm[0, 1] = 5.0
    pdm[0, 2] = 9.0
    pdm[0, 3] = 9.0
    pdm[0, 4] = 8.0
    pdm[1, 2] = 10.0
    pdm[1, 3] = 10.0
    pdm[1, 4] = 9.0
    pdm[2, 3] = 8.0
    pdm[2, 4] = 7.0
    pdm[3, 4] = 3.0
    pdm = pdm + pdm.T

    Q = peeler.compute_Q(pdm)
    Q1 = torch.tensor(
        [
            [0, -50, -38, -34, -34],
            [-50, 0, -38, -34, -34],
            [-38, -38, 0, -40, -40],
            [-34, -34, -40, 0, -48],
            [-34, -34, -40, -48, 0],
        ]
    ).double()
    assert torch.allclose(Q, Q1)


def test_nj_knownQ():
    pdm = torch.zeros((5, 5), requires_grad=True).double()
    pdm[0, 1] = 5.0
    pdm[0, 2] = 9.0
    pdm[0, 3] = 9.0
    pdm[0, 4] = 8.0
    pdm[1, 2] = 10.0
    pdm[1, 3] = 10.0
    pdm[1, 4] = 9.0
    pdm[2, 3] = 8.0
    pdm[2, 4] = 7.0
    pdm[3, 4] = 3.0
    pdm = pdm + pdm.T

    peel, blens = peeler.nj(pdm)
    peel1 = np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 2, 7], [7, 6, 8]])
    peel2 = np.allclose(peel, [[0, 1, 5], [5, 2, 6], [6, 3, 7], [7, 4, 8]])
    peel3 = np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 6, 7], [2, 7, 8]])
    assert peel1 or peel2 or peel3, "Wrong topology. NB. not all correct cases covered."
    assert torch.isclose(
        sum(blens).double(), torch.tensor(17).double()
    ), "Wrong sum of branch lengths."
    assert blens.requires_grad == True


def test_nj_soft():
    leaf_r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    pdm = Cutils.get_pdm_torch(leaf_r, leaf_dir)
    pdm.requires_grad = True
    for _ in range(1000):
        peel, blens = peeler.nj(pdm, tau=0.0001)

        peel_check = []
        peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
        peel_check.append(np.allclose(peel, [[2, 3, 4], [1, 4, 5], [0, 5, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 2, 5], [5, 3, 6]]))
        peel_check.append(np.allclose(peel, [[2, 3, 4], [0, 4, 5], [1, 5, 6]]))
        peel_check.append(np.allclose(peel, [[2, 3, 4], [0, 1, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 3, 5], [2, 5, 6]]))
        
        assert sum(
            peel_check
        ), f"Possibly an incorrect tree topology:\n{peel}"
        assert torch.isclose(
            sum(blens).float(), torch.tensor(2.0318).float(), atol=0.05
        ), "Incorrect total branch length"
        assert blens.requires_grad == True, "Branch lengths must carry gradients."


def test_soft_nj_knownQ():
    pdm = torch.zeros((5, 5), requires_grad=True).double()
    pdm[0, 1] = 5.0
    pdm[0, 2] = 9.0
    pdm[0, 3] = 9.0
    pdm[0, 4] = 8.0
    pdm[1, 2] = 10.0
    pdm[1, 3] = 10.0
    pdm[1, 4] = 9.0
    pdm[2, 3] = 8.0
    pdm[2, 4] = 7.0
    pdm[3, 4] = 3.0
    pdm = pdm + pdm.T

    for _ in range(1000):
        peel, blens = peeler.nj(pdm, 0.0001)
        peel_check = []
        peel_check.append(np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 2, 7], [7, 6, 8]]))
        peel_check.append(np.allclose(peel, [[0, 1, 5], [3, 4, 6], [2, 6, 7], [5, 7, 8]]))
        peel_check.append(np.allclose(peel, [[0, 1, 5], [5, 2, 6], [6, 3, 7], [7, 4, 8]]))
        peel_check.append(np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 6, 7], [2, 7, 8]]))
        peel_check.append(np.allclose(peel, [[0, 1, 5], [5, 2, 6], [6, 4, 7], [3, 7, 8]]))
        peel_check.append(np.allclose(peel, [[0, 1, 5], [5, 2, 6], [3, 4, 7], [6, 7, 8]]))
        peel_check.append(np.allclose(peel, [[1, 0, 5], [2, 5, 6], [4, 3, 7], [7, 6, 8]]))
        assert sum(peel_check), f"Probable incorrect tree topology: {peel}"
        assert torch.isclose(
            sum(blens).double(), torch.tensor(17).double(), atol=0.1
        ), "Wrong sum of branch lengths."


def test_soft_sort_1d():
    input = torch.tensor([2, 5.5, 3, -1, 0])
    permute = peeler.soft_sort(input.unsqueeze(-1).unsqueeze(0), tau=0.000001).squeeze()
    output = permute @ input
    correct = torch.tensor([5.5, 3, 2, 0, -1])
    assert torch.allclose(output, correct)


def test_soft_sort_2d():
    input = torch.tensor(
        [[0, 50, 38, 34], [50, 0, 38, 34], [38, 38, 0, 40], [34, 34, 40, 0]]
    )
    permute = peeler.soft_sort(input.unsqueeze(-1), tau=0.000001)

    row0_argmax = permute[0, 0]
    assert torch.allclose(row0_argmax[1], torch.ones(1))
    assert torch.allclose(sum(row0_argmax), torch.ones(1))
    row1_argmax = permute[1, 0]
    assert torch.allclose(row1_argmax[0], torch.ones(1))
    assert torch.allclose(sum(row1_argmax), torch.ones(1))
    row2_argmax = permute[2, 0]
    assert torch.allclose(row2_argmax[3], torch.ones(1))
    assert torch.allclose(sum(row2_argmax), torch.ones(1))


def test_soft_argmin_one_hot():
    input_2d = torch.tensor(([4, 5, 10], [3, 4, 2.3], [20, 2, 8]))
    one_hot_i, one_hot_j = peeler.soft_argmin_one_hot(input_2d, tau=0.000001)
    assert torch.allclose(one_hot_i, torch.tensor([0.0, 0.0, 1.0])), "wrong i index"
    assert torch.allclose(one_hot_j, torch.tensor([0.0, 1.0, 0.0])), "wrong j index"
