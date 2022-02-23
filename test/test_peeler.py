import dendropy
import numpy as np
import torch
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy import (
    Cutils,
    peeler,
    poincare,
    utils,
    Cpeeler,
    Chyp_torch,
    Chyp_np,
    node,
)
from dodonaphy.phylo import compress_alignment
from dodonaphy.vi import DodonaphyVI
from numpy import genfromtxt


def test_make_peel_geodesic_dogbone():
    leaf_r = np.array([0.5, 0.5, 0.5, 0.5])
    leaf_theta = np.array([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)
    peel, _ = peeler.make_hard_peel_geodesic(leaf_hyp)
    expected_peel = np.array([[1, 0, 4], [3, 2, 5], [4, 5, 6]])

    assert np.allclose(peel, expected_peel)


def test_make_peel_geodesic_example0():
    leaf_locs = np.array(
        [
            (0.05909264, 0.16842421, -0.03628194),
            (0.08532969, -0.07187002, 0.17884444),
            (-0.11422830, 0.01955054, 0.14127290),
            (-0.06550432, 0.07029946, -0.14566249),
            (-0.07060744, -0.12278600, -0.17569585),
            (0.11386343, -0.03121063, -0.18112418),
        ]
    )
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_locs)
    _, _ = peeler.make_hard_peel_geodesic(leaf_hyp)


def test_make_peel_geodesic_example1():
    leaf_locs = np.array(
        [
            [1.5330e-02, 1.0385e-02],
            [-6.0270e-02, 4.4329e-02],
            [-1.1253e-01, -1.5676e-01],
            [1.0916e-01, -7.2296e-02],
            [5.9408e-02, 3.0677e-04],
            [-4.0814e-02, -3.1838e-01],
        ]
    )
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_locs)
    peel, _ = peeler.make_hard_peel_geodesic(leaf_hyp)
    for i in range(5):
        assert int(peel[i][0]) is not int(peel[i][1])
        assert int(peel[i][0]) is not int(peel[i][2])
        assert int(peel[i][1]) is not int(peel[i][2])


def test_make_peel_geodesic_example2():
    leaf_locs = np.array(
        [
            [2.09999997e-02, -2.09410252e-01],
            [1.01784302e-01, 3.18292447e-02],
            [-9.41199092e-02, 7.48080376e-02],
            [-1.85000000e-02, -1.25415666e-01],
            [-3.39999999e-02, -1.52410744e-01],
            [-1.40397397e-02, 1.47278753e-01],
        ]
    )
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_locs)
    peel, _ = peeler.make_hard_peel_geodesic(leaf_hyp)
    for i in range(5):
        assert int(peel[i][0]) is not int(peel[i][1])
        assert int(peel[i][0]) is not int(peel[i][2])
        assert int(peel[i][1]) is not int(peel[i][2])


def test_nj():
    leaf_r = np.array([0.5, 0.5, 0.5, 0.5])
    leaf_theta = np.array([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)

    pdm = Chyp_np.get_pdm(leaf_hyp)
    peel, blens = peeler.nj_np(pdm)

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
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."
    assert np.isclose(sum(blens), 3.29274740, atol=0.001)


def test_nj_matsumoto():
    leaf_r = np.array([0.5, 0.5, 0.5, 0.5])
    leaf_theta = np.array([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)

    pdm = Chyp_np.get_pdm(leaf_hyp, matsumoto=True)
    peel, blens = peeler.nj_np(pdm)

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
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."
    assert np.isclose(sum(blens), 2.0318, atol=0.001)  # Using Matsumoto adjustment


def test_nj_dendropy():
    leaf_r = np.array([0.5, 0.5, 0.5, 0.5])
    leaf_theta = np.array([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)
    pdm = Chyp_np.get_pdm(leaf_hyp)

    peel_dendro, blens_dendro = peeler.nj_np(pdm)
    _, blens = Cpeeler.nj_np(pdm)

    peel_check = []
    peel_check.append(np.allclose(peel_dendro, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[2, 3, 4], [1, 4, 5], [0, 5, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[0, 1, 4], [4, 2, 5], [5, 3, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[2, 3, 4], [0, 4, 5], [1, 5, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[2, 3, 4], [0, 1, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel_dendro, [[0, 1, 4], [4, 3, 5], [2, 5, 6]]))
    assert sum(
        peel_check
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."

    assert sum(peel_check)
    assert np.allclose(sum(blens), sum(blens_dendro))


def test_nj_np():
    leaf_r = np.array([0.5, 0.5, 0.5, 0.5])
    leaf_theta = np.array([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)

    pdm = Chyp_np.get_pdm(leaf_hyp)
    peel, blens = peeler.nj_np(pdm)

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
    ), "Wrong topology. NB. non-exhaustive check of correct topologies."
    # assert np.isclose(sum(blens), 2.0318, atol=0.001)  # Using Matsumoto adjustment
    # assert np.isclose(sum(blens), 3.29274740, atol=0.001)


def test_nj_uneven():
    leaf_r = np.array([0.1, 0.2, 0.3, 0.4])
    leaf_theta = np.array([np.pi / 2, -np.pi / 10, np.pi, -np.pi * 6 / 8])
    leaf_dir = Cutils.angle_to_directional_np(leaf_theta)
    leaf_poin = Cutils.dir_to_cart_np(leaf_r, leaf_dir)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_poin)

    pdm = Chyp_np.get_pdm(leaf_hyp)
    peel, _ = peeler.nj_np(pdm)
    peel_check = []
    peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [1, 4, 5], [0, 5, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 2, 5], [5, 3, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 4, 5], [5, 3, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 2, 5], [3, 5, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 4, 5], [3, 5, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [0, 4, 5], [1, 5, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [0, 1, 5], [5, 4, 6]]))
    peel_check.append(np.allclose(peel, [[2, 3, 4], [0, 1, 5], [4, 5, 6]]))
    peel_check.append(np.allclose(peel, [[0, 1, 4], [4, 3, 5], [2, 5, 6]]))
    assert sum(
        peel_check
    ), f"Wrong topology. NB. non-exhaustive check of correct topologies. Peel: {peel}"


def test_compute_Q_dendropy():
    pdm = np.zeros((5, 5)).astype(np.double)
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

    n_pool = len(pdm)
    Q = np.zeros((n_pool, n_pool))

    # initialise node pool
    node_pool = [node.Node(taxon) for taxon in range(n_pool)]

    D = np.zeros_like(Q)
    # cache calculations
    for nd1 in node_pool:
        nd1._nj_xsub = 0.0
        for nd2 in node_pool:
            if nd1 is nd2:
                continue
            dist = pdm[nd1.taxon, nd2.taxon]
            D[nd1.taxon, nd2.taxon] = dist
            nd1._nj_distances[nd2.taxon] = dist
            nd1._nj_xsub += dist

    for idx1, nd1 in enumerate(node_pool[:-1]):
        for _, nd2 in enumerate(node_pool[idx1 + 1 :]):
            v1 = (n_pool - 2) * nd1._nj_distances[nd2.taxon]
            qvalue = v1 - nd1._nj_xsub - nd2._nj_xsub
            Q[nd1.taxon, nd2.taxon] = qvalue

    Q = Q + Q.T
    Q_actual = np.array(
        [
            [0, -50, -38, -34, -34],
            [-50, 0, -38, -34, -34],
            [-38, -38, 0, -40, -40],
            [-34, -34, -40, 0, -48],
            [-34, -34, -40, -48, 0],
        ]
    ).astype(np.double)
    assert np.allclose(Q, Q_actual)


test_compute_Q_dendropy()


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

    Q_test = peeler.compute_Q(pdm, fill_value=0)
    Q_actual = torch.tensor(
        [
            [0, -50, -38, -34, -34],
            [-50, 0, -38, -34, -34],
            [-38, -38, 0, -40, -40],
            [-34, -34, -40, 0, -48],
            [-34, -34, -40, -48, 0],
        ]
    ).double()
    assert torch.allclose(Q_test, Q_actual), f"Q_test: {Q_test}\n Q_actual: {Q_actual}"


def test_nj_soft():
    leaf_r = torch.tensor([0.5, 0.5, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi / 10, -np.pi / 10, np.pi * 6 / 8, -np.pi * 6 / 8])
    leaf_dir = utils.angle_to_directional(leaf_theta)
    leaf_locs = utils.dir_to_cart(leaf_r, leaf_dir).detach().numpy().astype(np.double)
    leaf_hyp = Chyp_np.poincare_to_hyper_2d(leaf_locs)

    pdm = torch.tensor(Chyp_np.get_pdm(leaf_hyp))
    pdm.requires_grad = True
    for i in range(10):
        peel, blens = peeler.nj_torch(pdm, tau=1e-7)

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
        ), f"Iteration: {i}. Possibly an incorrect tree topology:\n{peel}"
        # assert torch.isclose(
        #     sum(blens).float(), torch.tensor(2.0318).float(), atol=0.05
        # ), f"Iteration: {i}. Incorrect total branch length"
        assert torch.isclose(
            sum(blens).float(), torch.tensor(3.2927).float(), atol=0.05
        ), f"Iteration: {i}. Incorrect total branch length"
        assert blens.requires_grad == True, "Branch lengths must carry gradients."


def test_nj_soft_all_even():
    dists = torch.ones((6, 6), dtype=torch.double) - torch.eye(6, dtype=torch.double)
    peel, _ = peeler.nj_torch(dists, tau=1e-4)
    set1 = set(np.sort(np.unique(peel)))
    set2 = set(np.arange(11))
    assert set1 == set2, f"Not all nodes in peel: {peel}"


def test_nj_eg1():
    dists_1d = genfromtxt("./test/test_peel_data.txt", dtype=np.double, delimiter=", ")
    tril_idx = np.tril_indices(17, -1)
    dist_2d = np.zeros((17, 17), dtype=np.double)
    dist_2d[tril_idx[0], tril_idx[1]] = dists_1d
    dist_2d[tril_idx[1], tril_idx[0]] = dists_1d

    peel_hard, _ = peeler.nj_np(dist_2d)
    peel_soft, _ = peeler.nj_torch(torch.tensor(dist_2d, dtype=torch.double), tau=1e-18)
    children_hard = set((frozenset(peel_hard[i, :2]) for i in range(16)))
    children_soft = set((frozenset(peel_soft[i, :2]) for i in range(16)))
    assert (
        children_soft == children_hard
    ), f"Bad soft peel:\n{peel_soft}\nHard peel:\n{peel_hard}"


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

    for i in range(10):
        peel, blens = peeler.nj_torch(pdm, tau=1e-5)
        peel_check = []
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 2, 7], [7, 6, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [3, 4, 6], [2, 6, 7], [5, 7, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [5, 2, 6], [6, 3, 7], [7, 4, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [3, 4, 6], [5, 6, 7], [2, 7, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [5, 2, 6], [6, 4, 7], [3, 7, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[0, 1, 5], [5, 2, 6], [3, 4, 7], [6, 7, 8]])
        )
        peel_check.append(
            np.allclose(peel, [[1, 0, 5], [2, 5, 6], [4, 3, 7], [7, 6, 8]])
        )
        assert sum(peel_check), f"Probable incorrect tree topology: {peel}"
        assert torch.isclose(
            sum(blens).double(), torch.tensor(17).double(), atol=0.1
        ), f"Iteration{i}. Wrong sum of branch lengths: {sum(blens).double()} != 17"


def test_soft_sort_1d():
    input_arr = torch.tensor([2, 5.5, 3, -1, 0])
    permute = peeler.soft_sort(
        input_arr.unsqueeze(-1).unsqueeze(0), tau=0.000001
    ).squeeze()
    output = permute @ input_arr
    correct = torch.tensor([5.5, 3, 2, 0, -1])
    assert torch.allclose(output, correct)


def test_soft_sort_2d():
    input_arr = torch.tensor(
        [[0, 50, 38, 34], [50, 0, 38, 34], [38, 38, 0, 40], [34, 34, 40, 0]]
    )
    permute = peeler.soft_sort(input_arr.unsqueeze(-1), tau=0.000001)

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
    input_arr_2d = torch.tensor(([4, 5, 10], [3, 4, 2.3], [20, 2, 8]))
    one_hot_i, one_hot_j = peeler.soft_argmin_one_hot(input_arr_2d, tau=1e-4)
    assert torch.allclose(one_hot_i, torch.tensor([0.0, 0.0, 1.0])), "wrong i index"
    assert torch.allclose(one_hot_j, torch.tensor([0.0, 1.0, 0.0])), "wrong j index"


def test_geodesic():
    leaf_locs = 0.4 * np.array(
        [
            [np.cos(np.pi), np.sin(np.pi)],
            [np.cos(0.9 * np.pi), np.sin(0.9 * np.pi)],
            [np.cos(-0.5 * np.pi), np.sin(-0.5 * np.pi)],
        ]
    )
    peel, _ = peeler.make_hard_peel_geodesic(leaf_locs)

    peel_check = []
    peel_check.append(np.allclose(peel, [[1, 0, 3], [3, 2, 4]]))
    peel_check.append(np.allclose(peel, [[0, 1, 3], [3, 2, 4]]))
    peel_check.append(np.allclose(peel, [[1, 0, 3], [2, 3, 4]]))
    peel_check.append(np.allclose(peel, [[0, 1, 3], [2, 3, 4]]))
    assert sum(peel_check), f"Incorrect geodesic peel: {peel}"


def test_soft_geodesic0():
    leaf_locs_poin = 0.9 * torch.tensor(
        [
            [np.cos(np.pi), np.sin(np.pi)],
            [np.cos(0.9 * np.pi), np.sin(0.9 * np.pi)],
            [np.cos(-0.5 * np.pi), np.sin(-0.5 * np.pi)],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    leaf_locs_hyp = Chyp_torch.poincare_to_hyper(leaf_locs_poin)
    peel, int_locs, _ = peeler.make_soft_peel_tips(
        leaf_locs_poin, connector="geodesics", curvature=-torch.ones(1)
    )
    _, int_locs1 = peeler.make_hard_peel_geodesic(leaf_locs_hyp.detach().numpy())
    peel_check = []
    peel_check.append(np.allclose(peel, [[1, 0, 3], [3, 2, 4]]))
    peel_check.append(np.allclose(peel, [[0, 1, 3], [3, 2, 4]]))
    peel_check.append(np.allclose(peel, [[1, 0, 3], [2, 3, 4]]))
    peel_check.append(np.allclose(peel, [[0, 1, 3], [2, 3, 4]]))
    int_locs1_poin = np.zeros((2, 2))
    for i in range(2):
        int_locs1_poin[i] = Chyp_torch.hyper_to_poincare(int_locs1[i])
    assert sum(peel_check), f"Incorrect peel: {peel}"
    assert np.allclose(
        int_locs.detach().numpy(), int_locs1_poin
    ), f"{int_locs} != {int_locs1}"


def test_soft_geodesic1():
    leaf_r = torch.tensor([0.6, 0.6, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi * 0.2, 0, np.pi, -np.pi * 0.9])
    leaf_dir = utils.angle_to_directional(leaf_theta)
    leaf_locs = Cutils.dir_to_cart_np(leaf_r, leaf_dir).requires_grad_(True)
    for i in range(10):
        peel, int_locs, blens = peeler.make_soft_peel_tips(
            leaf_locs, connector="geodesics", curvature=-torch.ones(1)
        )

        peel_check = []
        peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [4, 5, 6]]))
        peel_check.append(np.allclose(peel, [[1, 0, 4], [3, 2, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[1, 0, 4], [2, 3, 5], [4, 5, 6]]))
        peel_check.append(np.allclose(peel, [[1, 0, 4], [2, 3, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [2, 3, 5], [4, 5, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [3, 2, 5], [5, 4, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [3, 2, 5], [4, 5, 6]]))
        peel_check.append(np.allclose(peel, [[0, 1, 4], [3, 2, 5], [4, 5, 6]]))
        assert sum(peel_check), f"Iteration {i}. Possibly incorrect peel: {peel}"
        assert int_locs.requires_grad == True
        assert blens.requires_grad == True


def test_soft_geodesic_optim():
    leaf_r = torch.tensor([0.8, 0.8, 0.5, 0.5])
    leaf_theta = torch.tensor([np.pi / 4, -np.pi / 7, np.pi * 7 / 10, -np.pi * 9 / 10])
    leaf_dir = utils.angle_to_directional(leaf_theta)
    params = {"leaf_locs": utils.dir_to_cart(leaf_r, leaf_dir).requires_grad_(True)}
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=4
    )
    dna = simulate_discrete_chars(
        seq_len=100, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyVI(partials, weights, dim=2, embedder="up", connector="geodesics")
    optimizer = torch.optim.Adam(list(params.values()), lr=1)
    optimizer.zero_grad()
    loss = -mymod.elbo_siwae(1)
    loss.backward()
    optimizer.step()


def test_hyp_lca_grad():
    from_loc = torch.tensor([0.1, 0.1], dtype=torch.double, requires_grad=True)
    to_loc = torch.tensor([0.1, -0.1], dtype=torch.double, requires_grad=True)
    params = {"from_loc": from_loc, "to_loc": to_loc}
    optimizer = torch.optim.Adam(list(params.values()), lr=1)
    optimizer.zero_grad()
    loss = poincare.hyp_lca(from_loc, to_loc, return_coord=False)
    loss.backward()
    optimizer.step()
