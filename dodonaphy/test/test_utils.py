import pytest
import torch
import numpy as np
from pytest import approx
import ete3

from src.utils import utilFunc


def test_make_peel_simple():
    # Connect three evenly spaced leaves
    # It seems this is only coming about when S=3
    # Issue coming from utils.py#L371
    S = 3
    leaf_r = .5*torch.ones(S)
    leaf_theta = torch.tensor([np.pi/6, 0., -np.pi/6])
    leaf_dir = utilFunc.angle_to_directional(leaf_theta)

    # internal node with angle in between nodes 0 and 1
    int_r = torch.tensor([.25])
    int_theta = torch.tensor([np.pi/12])
    int_dir = utilFunc.angle_to_directional(int_theta)

    # Connect nodes
    peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r,
                              int_dir)

    # See utilFunc.plot_tree:
    # Tree should connect 0 and 1 to internal node 3
    # root node 4, should connect to 0 and 3.
    assert np.allclose(peel, np.array([[1, 2, 3], [0, 3, 4]]))


def test_make_peel_dogbone():
    # Take 4 leaves and form a dogbone tree
    leaf_r = torch.tensor([.5, .5, .8, .8])
    leaf_theta = torch.tensor([np.pi/6, 0., -np.pi*.7, -np.pi*.8])
    leaf_dir = utilFunc.angle_to_directional(leaf_theta)

    int_r = torch.tensor([.25, .4])
    int_theta = torch.tensor([np.pi/12, -np.pi*.75])
    int_dir = utilFunc.angle_to_directional(int_theta)

    # make a tree
    peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r,
                              int_dir)

    assert np.allclose(peel, np.array([[2, 3, 5],
                                       [1, 5, 4],
                                       [0, 4, 6]]))


def test_make_peel_first_leaf_connection():
    leaf_r = torch.tensor([.5, .5, .5, .5])
    leaf_theta = torch.tensor([0., np.pi / 6, -np.pi / 6, np.pi])
    leaf_dir = utilFunc.angle_to_directional(leaf_theta)

    int_r = torch.tensor([.3, 0])
    int_theta = torch.tensor([np.pi/6, 0])
    int_dir = utilFunc.angle_to_directional(int_theta)

    # make a tree
    peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)

    assert np.allclose(peel, np.array([[3, 2, 5],
                                      [1, 5, 4],
                                      [0, 4, 6]]))


def test_hyperbolic_distance():
    r1 = torch.tensor([0.3])
    r2 = torch.tensor([.6])
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(-.5), torch.as_tensor(np.sqrt(0.75))])
    dist = utilFunc.hyperbolic_distance(
        r1, r2, dir1,
        dir2, torch.tensor([1.]))
    assert 1.438266 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_boundary():
    r1 = torch.tensor([0.])
    r2 = torch.tensor([.9999999])
    dir1 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dir2 = torch.tensor([torch.as_tensor(0.), torch.as_tensor(1.)])
    dist = utilFunc.hyperbolic_distance(
        r1, r2, dir1,
        dir2, torch.tensor([1.]))
    assert 16.635532 == pytest.approx(dist.item(), 0.0001)


def test_hyperbolic_distance_zero():
    dir1 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dir2 = torch.tensor([torch.as_tensor(1./np.sqrt(2)), torch.as_tensor(1./np.sqrt(2))])
    dist = utilFunc.hyperbolic_distance(
        torch.tensor([0.5]), torch.tensor([0.5]), dir1,
        dir2, torch.tensor([1.]))
    assert 0. == pytest.approx(dist.item(), abs=0.05)


def test_hydra_2d_compare_output():

    D = np.ones((3, 3), float)
    np.fill_diagonal(D, 0.0)
    dim = 2
    emm = utilFunc.hydra(D, dim)

    # Compare to output of hydra in r
    assert emm['curvature'] == approx(1)
    assert emm['dim'] == approx(dim)
    assert emm['directional'] == approx(np.array([
        [0.8660254038, -0.5000000000],
        [-0.8660254038, -0.5000000000],
        [0.0000000000, 1.0000000000]
    ]))
    assert emm['r'] == approx(
        np.array([0.2182178902, 0.2182178902, 0.2182178902]))
    assert emm['theta'] == approx(
        np.array([-0.5235987756, -2.6179938780, 1.5707963268]))


def test_hydra_3d_compare_output():

    D = np.array([[0, 1, 2.5, 3, 1], [1, 0, 2.2, 1, 2], [
                 2.5, 2.2, 0, 3, 1], [3, 1, 3, 0, 2], [1, 2, 1, 2, 0]])
    dim = 3
    emm = utilFunc.hydra(D, dim, equi_adj=0)

    # Compare to output of hydra in r
    # NB: some directions reversed from r due to opposite eigenvectors
    assert emm['curvature'] == approx(1)
    assert emm['dim'] == approx(dim)
    assert emm['r'] == approx(np.array(
        [0.6274604254, 0.2705702432, 0.6461609880, 0.6779027826, 0.2182178902])
    )
    assert emm['directional'] == approx(np.array([
        [0.0821320214, -0.7981301820, 0.5968605730],
        [-0.5711268681, -0.6535591501, -0.4966634050],
        [-0.2070516064, 0.6620783581, 0.7202651457],
        [0.0811501819, 0.1418186867, -0.9865607473],
        [0.8008637619, 0.2731045145, 0.5329457375]
    ]))


def test_all_pairwise_distance_ete3():
    t = ete3.Tree('(A:1,(B:1,(C:1,D:1):0.5):0.5);')
    nodes = t.get_tree_root().get_descendants()
    dist = [t.get_distance(x, y) for x in nodes for y in nodes]
    dist = np.array(dist).reshape(len(nodes), len(nodes))
    print(dist)


def test_dir_to_cart_1d():
    u = torch.tensor(10.)
    r = torch.norm(u)
    directional = u / r
    loc = utilFunc.dir_to_cart(r, directional)
    assert loc == approx(u)


def test_dir_to_cart_5d():
    u = torch.tensor((10., 2.3, 43., -4., 4.5))
    r = torch.norm(u)
    directional = u / r
    loc = utilFunc.dir_to_cart(r, directional)
    assert loc == approx(u)


# def test_make_peel_geodesic():
#     leaf_locs = torch.tensor([
#         [0.06644704, 0.18495312, -0.03839615],
#         [-0.12724270, -0.02395994, 0.20075329]])
#     lca = utilFunc.hyp_lca(leaf_locs[0], leaf_locs[1])
#     d0_1 = utilFunc.hyperbolic_distance_locs(leaf_locs[0], leaf_locs[1])
#     d0_lca = utilFunc.hyperbolic_distance_locs(leaf_locs[0], lca)
#     d1_lca = utilFunc.hyperbolic_distance_locs(leaf_locs[1], lca)
#     d0_lca_d1 = d0_lca + d1_lca
#     # // d0_1 should be the same as d0_lca_d1
#     assert pytest.approx(d0_1, d0_lca_d1)

#     leaf_locs = torch.tensor([
#         (0.05909264, 0.16842421, -0.03628194),
#         (0.08532969, -0.07187002, 0.17884444),
#         (-0.11422830, 0.01955054, 0.14127290),
#         (-0.06550432, 0.07029946, -0.14566249),
#         (-0.07060744, -0.12278600, -0.17569585),
#         (0.11386343, -0.03121063, -0.18112418)])

#     # d01 = torch.log(torch.cosh(utilFunc.hyperbolic_distance_locs(leaf_locs[0], leaf_locs[1])))
#     # d02 = torch.log(torch.cosh(utilFunc.hyperbolic_distance_locs(leaf_locs[0], leaf_locs[2])))
#     # d03 = torch.log(torch.cosh(utilFunc.hyperbolic_distance_locs(leaf_locs[0], leaf_locs[3])))

#     peel, int_locs = utilFunc.make_peel_geodesics(leaf_locs)
#     print(peel)

#     import matplotlib.pyplot as plt
#     X = torch.cat((leaf_locs, int_locs), dim=0)
#     ax = plt.subplot(1, 1, 1)
#     utilFunc.plot_tree(ax, peel, X)
#     plt.show()


# def test_make_peel_geodesic_dogbone():

#     leaf_r = torch.tensor([.5, .5, .8, .8])
#     leaf_theta = torch.tensor([np.pi/6, 0., -np.pi*.7, -np.pi*.8])
#     leaf_dir = utilFunc.angle_to_directional(leaf_theta)
#     leaf_locs = utilFunc.dir_to_cart(leaf_r, leaf_dir)

#     peel, int_locs = utilFunc.make_peel_geodesics(leaf_locs)
#     X = torch.cat((leaf_locs, int_locs), dim=0)
#     print("\npeel")
#     print(peel)
#     print("locs")
#     print(X.detach().numpy())

#     import matplotlib.pyplot as plt
#     X = torch.cat((leaf_locs, int_locs), dim=0)
#     ax = plt.subplot(1, 1, 1)
#     utilFunc.plot_tree(ax, peel, X)
#     plt.show()

#     assert np.allclose(peel, np.array([[2, 3, 5],
#                                        [1, 5, 4],
#                                        [0, 4, 6]]))
