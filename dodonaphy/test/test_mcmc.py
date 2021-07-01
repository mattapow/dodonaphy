import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
import random
import numpy as np

from src.mcmc import DodonaphyMCMC as mcmc
from src.phylo import compress_alignment


def test_mcmc_incentre():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}

    # MCMC parameters
    step_scale = .1
    nChains = 1
    connect_method = 'incentre'  # 'incentre', 'geodesics' or 'mst'
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior['birth_rate'], death_rate=prior['death_rate'], num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

    partials, weights = compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(dim, partials[:], weights, dists, path_write_mcmc,
             epochs=epochs, step_scale=step_scale, burnin=burnin,
             nChains=nChains, connect_method=connect_method, **prior)


def test_mcmc_mst():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}

    # MCMC parameters
    step_scale = .1
    nChains = 1
    connect_method = 'mst'  # 'incentre', 'geodesics' or 'mst'
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior['birth_rate'], death_rate=prior['death_rate'], num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

    partials, weights = compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(dim, partials[:], weights, dists, path_write_mcmc,
             epochs=epochs, step_scale=step_scale, burnin=burnin,
             n_grids=1, n_trials=1, nChains=nChains,
             connect_method=connect_method, **prior)


def test_mcmc_geodesics():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}

    # MCMC parameters
    step_scale = .1
    nChains = 1
    connect_method = 'geodesics'  # 'incentre', 'geodesics' or 'mst'
    burnin = 0
    epochs = 1

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=prior['birth_rate'], death_rate=prior['death_rate'], num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

    partials, weights = compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # Run Dodoanphy MCMC
    path_write_mcmc = None
    mcmc.run(dim, partials[:], weights, dists, path_write_mcmc,
             epochs=epochs, step_scale=step_scale, burnin=burnin,
             nChains=nChains, connect_method=connect_method, **prior)
