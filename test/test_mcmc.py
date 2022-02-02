import random

import dendropy
import numpy as np
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim
from dodonaphy.mcmc import DodonaphyMCMC as mcmc
from dodonaphy import Cphylo
from dodonaphy.utils import tip_distances


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

    partials, weights = Cphylo.compress_alignment_np(dna)

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
    dim = 2  # number of dimensions for embedding
    S = 11  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2.0, "death_rate": 0.5}

    # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=prior["birth_rate"],
        death_rate=prior["death_rate"],
        num_extant_tips=S,
    )
    dna = simulate_discrete_chars(
        seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69()
    )
    partials, weights = Cphylo.compress_alignment_np(dna)

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

    partials, weights = Cphylo.compress_alignment_np(dna)

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

    partials, weights = Cphylo.compress_alignment_np(dna)

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
