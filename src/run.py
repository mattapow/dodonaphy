import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
import random
import os
import numpy as np

from src.vi import DodonaphyVI as vi
from src.mcmc import DodonaphyMCMC as mcmc
from src.phylo import compress_alignment


def main():

    dim = 2  # number of dimensions for embedding
    S = 12  # number of sequences to simulate
    L = 1000  # length of sequences to simulate

    # simulate a tree
    rng = random.Random(1)
    prior = {"birth_rate": 2., "death_rate": .5}
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
    # TODO: should distances be genetic distance or true tree patristic distance?

    # VI parameters
    epochs = 10000
    k_samples = 10
    n_draws = 10
    boosts = 3
    init_trials = 100
    init_grids = 10

    # Make experiment folder
    path_write = "./data/Taxa%dDim%dBoosts%d" % (S, dim, boosts)
    os.makedirs(path_write, exist_ok=False)

    # save dna to nexus
    dna.write_to_path(path_write + "/dna.nex", "nexus")

    # Run Dodonaphy variational inference
    path_write_vi = os.path.abspath(os.path.join(path_write, "vi"))
    os.mkdir(path_write_vi)
    vi.run(dim, S, partials[:], weights, dists, path_write_vi,
           epochs=epochs, k_samples=k_samples, n_draws=n_draws, boosts=boosts,
           init_grids=init_grids, init_trials=init_trials, **prior)

    # Run Dodoanphy MCMC
    path_write_mcmc = os.path.abspath(os.path.join(path_write, "mcmc"))
    os.mkdir(path_write_mcmc)
    mcmc.run(dim, partials[:], weights, dists, path_write_mcmc, epochs=2, step_scale=0.01, save_period=1, **prior)

    # Make folder for BEAST
    path_write_beast = os.path.abspath(os.path.join(path_write, "beast"))
    os.mkdir(path_write_beast)


if __name__ == "__main__":
    main()
