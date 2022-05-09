from dodonaphy.chain import Chain
import numpy as np

def test_prior_normal():
    dim = 3
    leaf_x = np.random.randn(5, dim)
    chain = Chain(
        partials=[np.ones((5, 1, 1))],
        weights=np.ones(1, dtype=int),
        dim=dim,
        leaf_x=leaf_x,
        loss_fn="none",
        prior='normal',
    )
    ln_prior = chain.get_prior()
    assert ln_prior != float("Nan")


def test_prior_gammadir():
    dim = 3
    leaf_x = np.random.randn(5, dim)
    chain = Chain(
        partials=[np.ones((5, 1, 1))],
        weights=np.ones(1, dtype=int),
        dim=dim,
        leaf_x=leaf_x,
        loss_fn="none",
        prior='gammadir',
    )
    chain.blens = np.random.exponential(0.1, 5)
    ln_prior = chain.get_prior()
    assert ln_prior != float("Nan")
