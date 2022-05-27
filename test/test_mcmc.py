import random

import dendropy
import numpy as np
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.mcmc import DodonaphyMCMC as mcmc
from dodonaphy import phylo
from dodonaphy.utils import tip_distances
from pytest import approx


def test_mcmc_geodesics():
    # Simulation parameters
    dim = 2
    S = 6
    L = 1000
    prior = {"birth_rate": 2.0, "death_rate": 0.5}

    # MCMC parameters
    step_scale = 0.1
    n_chains = 1
    connector = "geodesics"
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior["birth_rate"],
        death_rate=prior["death_rate"],
        num_extant_tips=S,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng
    )

    partials, weights = phylo.compress_alignment(dna)

    dists = tip_distances(simtree, S)

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(
        dim,
        partials[:],
        weights,
        dists,
        path_write_mcmc,
        epochs=epochs,
        step_scale=step_scale,
        burnin=burnin,
        n_chains=n_chains,
        connector=connector,
    )


def test_mcmc_geodesics_wrap():
    # Simulation parameters
    dim = 3  # number of dimensions for embedding
    S = 11  # number of sequences to simulate

    # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2.0,
        death_rate=0.5,
        num_extant_tips=S,
    )
    dna = simulate_discrete_chars(
        seq_len=1000, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = phylo.compress_alignment(dna)
    dists = tip_distances(simtree, S)

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(
        dim,
        partials[:],
        weights,
        dists,
        path_write_mcmc,
        epochs=3,
        step_scale=0.1,
        burnin=0,
        n_chains=1,
        connector="geodesics",
        embedder="wrap",
        curvature=-2.0,
    )


def test_mcmc_project_up_nj():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2.0, "death_rate": 0.5}

    # MCMC parameters
    step_scale = 0.1
    n_chains = 1
    connector = "nj"
    embedder = "up"
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior["birth_rate"],
        death_rate=prior["death_rate"],
        num_extant_tips=S,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng
    )

    partials, weights = phylo.compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i + 1 :]):
            dists[i][i + j + 1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(
        dim,
        partials[:],
        weights,
        dists,
        path_write_mcmc,
        epochs=epochs,
        step_scale=step_scale,
        burnin=burnin,
        n_chains=n_chains,
        connector=connector,
        embedder=embedder,
        warm_up=0,
    )

def test_mcmc_project_up_nj_euclid():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2.0, "death_rate": 0.5}

    # MCMC parameters
    step_scale = 0.1
    n_chains = 1
    connector = "nj"
    embedder = "up"
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior["birth_rate"],
        death_rate=prior["death_rate"],
        num_extant_tips=S,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng
    )

    partials, weights = phylo.compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i + 1 :]):
            dists[i][i + j + 1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(
        dim,
        partials[:],
        weights,
        dists,
        path_write_mcmc,
        epochs=epochs,
        step_scale=step_scale,
        burnin=burnin,
        n_chains=n_chains,
        connector=connector,
        embedder=embedder,
        warm_up=0,
        curvature=0,
    )

def test_mcmc_wrap_nj():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2.0, "death_rate": 0.5}

    # MCMC parameters
    step_scale = 1e-10
    n_chains = 1
    connector = "nj"
    embedder = "wrap"
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior["birth_rate"],
        death_rate=prior["death_rate"],
        num_extant_tips=S,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng
    )

    partials, weights = phylo.compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i + 1 :]):
            dists[i][i + j + 1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(
        dim,
        partials[:],
        weights,
        dists,
        path_write_mcmc,
        epochs=epochs,
        step_scale=step_scale,
        burnin=burnin,
        n_chains=n_chains,
        connector=connector,
        embedder=embedder,
    )


def test_swap():
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=2.,
        death_rate=0.5,
        num_extant_tips=4,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=400, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng
    )
    partials, weights = phylo.compress_alignment(dna)
    mymcmc = mcmc(partials, weights, dim=3, n_chains=2)
    mymcmc.chains[0].ln_p = -1e10
    mymcmc.chains[0].ln_prior = 0
    mymcmc.chains[1].ln_p = 1
    mymcmc.chains[1].ln_prior = 2
    mymcmc.swap()
    assert mymcmc.chains[0].ln_p == approx(1)
    assert mymcmc.chains[0].ln_prior == approx(2)
    assert mymcmc.chains[0].chain_temp == approx(1)
    assert mymcmc.chains[1].ln_p == approx(-1e10)
    assert mymcmc.chains[1].ln_prior == approx(0)
    assert mymcmc.chains[1].chain_temp == approx(1.0 / (1 + 0.1))
