from src import utils, poincare, peeler
import pytest
import torch
import numpy as np


def test_make_peel_simple():
    # Connect three evenly spaced leaves
    # It seems this is only coming about when S=3
    # Issue coming from utils.py#L371
    S = 3
    leaf_r = .5*torch.ones(S)
    leaf_theta = torch.tensor([np.pi/6, 0., -np.pi/9])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    # internal node with angle in between nodes 0 and 1
    int_r = torch.tensor([.25])
    int_theta = torch.tensor([np.pi/12])
    int_dir = utils.angle_to_directional(int_theta)

    # Connect nodes
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    # See utilFunc.plot_tree:
    # Tree should connect 0 and 1 to internal node 3
    # root node 4, should connect to 0 and 3.
    assert np.allclose(peel, np.array([[1, 2, 3], [0, 3, 4]]))


def test_make_peel_dogbone():
    # Take 4 leaves and form a dogbone tree
    leaf_r = torch.tensor([.5, .5, .8, .8])
    leaf_theta = torch.tensor([np.pi/6, 0., -np.pi*.7, -np.pi*.8])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    int_r = torch.tensor([.25, .4])
    int_theta = torch.tensor([np.pi/12, -np.pi*.75])
    int_dir = utils.angle_to_directional(int_theta)

    # make a tree
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    assert np.allclose(peel, np.array([[2, 3, 5],
                                       [1, 5, 4],
                                       [0, 4, 6]]))


def test_make_peel_first_leaf_connection():
    leaf_r = torch.tensor([.5, .5, .5, .5])
    leaf_theta = torch.tensor([0., np.pi / 6, -np.pi / 6, np.pi])
    leaf_dir = utils.angle_to_directional(leaf_theta)

    int_r = torch.tensor([.3, 0])
    int_theta = torch.tensor([np.pi/6, 0])
    int_dir = utils.angle_to_directional(int_theta)

    # make a tree
    peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    correct1 = np.allclose(peel, np.array([[3, 2, 5],
                                           [1, 5, 4],
                                           [0, 4, 6]]))
    correct2 = np.allclose(peel, np.array([[2, 3, 5],
                                           [1, 5, 4],
                                           [0, 4, 6]]))

    assert correct1 or correct2


def test_make_peel_example1():
    leaf_locs = torch.tensor([[5.5330e-02, 4.0385e-02],
                             [6.0270e-02, 4.4329e-02],
                             [-1.1253e-01, -1.5676e-01],
                             [1.0916e-01, -7.2296e-02],
                             [5.9408e-02, 4.0677e-02],
                             [-4.0814e-02, -3.1838e-01]])

    int_locs = torch.tensor([[0.0352, 0.0405],
                             [0.0437, 0.0144],
                             [0.0375, -0.070],
                             [-0.0633, -0.1595]], dtype=torch.float64)

    leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
    int_r, int_dir = utils.cart_to_dir(int_locs)
    _ = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)


def test_make_peel_incentre():
    leaf_locs = torch.tensor([
        [0.06644704, 0.18495312, -0.03839615],
        [-0.12724270, -0.02395994, 0.20075329]])
    lca = poincare.hyp_lca(leaf_locs[0], leaf_locs[1])
    d0_1 = utils.hyperbolic_distance_locs(leaf_locs[0], leaf_locs[1])
    d0_lca = utils.hyperbolic_distance_locs(leaf_locs[0], lca)
    d1_lca = utils.hyperbolic_distance_locs(leaf_locs[1], lca)
    d0_lca_d1 = d0_lca + d1_lca
    # // d0_1 should be the same as d0_lca_d1
    assert pytest.approx(d0_1, d0_lca_d1)


def test_make_peel_geodesic_dogbone():

    leaf_r = torch.tensor([.5, .5, .5, .5])
    leaf_theta = torch.tensor([np.pi/10, -np.pi/10, np.pi*6/8, -np.pi*6/8])
    leaf_dir = utils.angle_to_directional(leaf_theta)
    leaf_locs = utils.dir_to_cart(leaf_r, leaf_dir)

    peel, int_locs = peeler.make_peel_incentre(leaf_locs)

    # import matplotlib.pyplot as plt
    # ax = plt.subplot(1, 1, 1)
    # X = np.concatenate((leaf_locs, int_locs))
    # tree.plot_tree(ax, peel, X)
    # plt.show()

    expected_peel = np.array([[1, 0, 4],
                             [3, 2, 5],
                             [4, 5, 6]])

    assert np.allclose(peel, expected_peel)


def test_make_peel_geodesic_example0():
    leaf_locs = torch.tensor([
        (0.05909264, 0.16842421, -0.03628194),
        (0.08532969, -0.07187002, 0.17884444),
        (-0.11422830, 0.01955054, 0.14127290),
        (-0.06550432, 0.07029946, -0.14566249),
        (-0.07060744, -0.12278600, -0.17569585),
        (0.11386343, -0.03121063, -0.18112418)])

    peel, int_locs = peeler.make_peel_incentre(leaf_locs)


def test_make_peel_geodesic_example1():
    # Example
    # Note that the LCA may extend the geodesic beyond either node

    leaf_locs = torch.tensor([[1.5330e-02, 1.0385e-02],
                             [-6.0270e-02, 4.4329e-02],
                             [-1.1253e-01, -1.5676e-01],
                             [1.0916e-01, -7.2296e-02],
                             [5.9408e-02, 3.0677e-04],
                             [-4.0814e-02, -3.1838e-01]])

    peel, int_locs = peeler.make_peel_incentre(leaf_locs)

    # import matplotlib.pyplot as plt
    # ax = plt.subplot(1, 1, 1)
    # X = np.concatenate((leaf_locs, int_locs))
    # tree.plot_tree(ax, peel, X)
    # plt.show()

    for i in range(5):
        assert(int(peel[i][0]) is not int(peel[i][1]))
        assert(int(peel[i][0]) is not int(peel[i][2]))
        assert(int(peel[i][1]) is not int(peel[i][2]))


def test_make_peel_geodesic_example2():
    leaf_locs = torch.tensor([[2.09999997e-02, -2.09410252e-01],
                              [1.01784302e-01, 3.18292447e-02],
                              [-9.41199092e-02, 7.48080376e-02],
                              [-1.85000000e-02, -1.25415666e-01],
                              [-3.39999999e-02, -1.52410744e-01],
                              [-1.40397397e-02, 1.47278753e-01]])

    # norm = torch.norm(leaf_locs, dim=1).unsqueeze(1)
    # leaf_locs = .3 * leaf_locs / norm

    peel, int_locs = peeler.make_peel_incentre(leaf_locs)

    # import matplotlib.pyplot as plt
    # ax = plt.subplot(1, 1, 1)
    # X = np.concatenate((leaf_locs, int_locs))
    # tree.plot_tree(ax, peel, X)
    # plt.show()

    # compare to previous run
    # assert np.allclose(peel, np.array([[4, 3, 6],
    #                                    [2, 5, 7],
    #                                    [0, 6, 8],
    #                                    [1, 8, 9],
    #                                    [7, 9, 10]]))

    # assert np.allclose(int_locs.detach().numpy(),
    #                    np.array([[-0.01653543, -0.12316275],
    #                             [-0.07874484, 0.08843125],
    #                             [0.04499456, -0.09277647],
    #                             [0.07194225, -0.0327512],
    #                             [0.01213983, 0.01511928]]))

    for i in range(5):
        assert(int(peel[i][0]) is not int(peel[i][1]))
        assert(int(peel[i][0]) is not int(peel[i][2]))
        assert(int(peel[i][1]) is not int(peel[i][2]))
