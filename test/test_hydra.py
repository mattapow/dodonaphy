import dodonaphy.hydra
import dodonaphy.hydraPlus
import numpy as np
from pytest import approx


def test_hydraPlus():
    # dists from birth (2.0) death (0.5) tree
    dists = np.array(
        [
            [0.0, 0.82291866, 0.82291866, 0.99150298, 0.82291866],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.99150298, 0.99150298, 0.99150298, 0.0, 0.99150298],
            [0.82291866, 0.1656876, 0.1656876, 0.99150298, 0.0],
        ]
    )
    dim = 2
    hp_obj = dodonaphy.hydraPlus.HydraPlus(dists, dim)
    stress_emm = hp_obj.embed(equi_adj=0.0, stress=True)
    assert stress_emm["stress_hydraPlus"] < stress_emm["stress_hydra"]


def test_stress_gradient():
    dists = np.array(
        [
            [0.0, 0.82291866, 0.82291866, 0.99150298, 0.82291866],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.99150298, 0.99150298, 0.99150298, 0.0, 0.99150298],
            [0.82291866, 0.1656876, 0.1656876, 0.99150298, 0.0],
        ]
    )
    loc = np.array(
        [
            [-0.30701093, -0.05559487],
            [0.08942463, 0.19905347],
            [0.08942463, 0.19905347],
            [0.16894449, -0.31915586],
            [0.0893763, 0.20177165],
        ]
    )
    dim = 2
    hp_obj = dodonaphy.hydraPlus.HydraPlus(dists, dim, curvature=-1.0)
    stress_grad = hp_obj.get_stress_gradient(loc.flatten())
    stress_grad_true = np.array(
        [
            [2.4939401, 0.6992880],
            [-0.4659294, -0.9865152],
            [-0.4659294, -0.9865152],
            [-1.1476661, 3.0800397],
            [-0.4282155, -1.9410802],
        ]
    ).flatten()
    assert stress_grad == approx(stress_grad_true)


def test_get_stress():
    dists = np.array(
        [
            [0.0, 0.82291866, 0.82291866, 0.99150298, 0.82291866],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.82291866, 0.0, 0.0, 0.99150298, 0.1656876],
            [0.99150298, 0.99150298, 0.99150298, 0.0, 0.99150298],
            [0.82291866, 0.1656876, 0.1656876, 0.99150298, 0.0],
        ]
    )
    loc = np.array(
        [
            [-0.30701093, -0.05559487],
            [0.08942463, 0.19905347],
            [0.08942463, 0.19905347],
            [0.16894449, -0.31915586],
            [0.0893763, 0.20177165],
        ]
    )
    hp_obj = dodonaphy.hydraPlus.HydraPlus(dists, dim=2)
    stress = hp_obj.get_stress(loc)
    stress_true = 1.31283560338539
    assert stress == approx(stress_true)


def test_hydra_2d_compare_output():

    D = np.ones((3, 3), float)
    np.fill_diagonal(D, 0.0)
    dim = 2
    emm = dodonaphy.hydra.hydra(D, dim)

    # Compare to output of hydra in r
    assert emm["curvature"] == approx(-1)
    assert emm["dim"] == approx(dim)
    assert emm["directional"] == approx(
        np.array(
            [
                [0.8660254038, -0.5000000000],
                [-0.8660254038, -0.5000000000],
                [0.0000000000, 1.0000000000],
            ]
        )
    )
    assert emm["r"] == approx(np.array([0.2182178902, 0.2182178902, 0.2182178902]))
    assert emm["theta"] == approx(
        np.array([-0.5235987756, -2.6179938780, 1.5707963268])
    )


def test_hydra_3d_compare_output():

    D = np.array(
        [
            [0, 1, 2.5, 3, 1],
            [1, 0, 2.2, 1, 2],
            [2.5, 2.2, 0, 3, 1],
            [3, 1, 3, 0, 2],
            [1, 2, 1, 2, 0],
        ]
    )
    dim = 3
    emm = dodonaphy.hydra.hydra(D, dim, equi_adj=0)

    # Compare to output of hydra in r
    # NB: some directions reversed from r due to opposite eigenvectors
    assert emm["curvature"] == approx(-1)
    assert emm["dim"] == approx(dim)
    assert emm["r"] == approx(
        np.array([0.6274604254, 0.2705702432, 0.6461609880, 0.6779027826, 0.2182178902])
    )
    assert emm["directional"] == approx(
        np.array(
            [
                [0.0821320214, -0.7981301820, 0.5968605730],
                [-0.5711268681, -0.6535591501, -0.4966634050],
                [-0.2070516064, 0.6620783581, 0.7202651457],
                [0.0811501819, 0.1418186867, -0.9865607473],
                [0.8008637619, 0.2731045145, 0.5329457375],
            ]
        )
    )
