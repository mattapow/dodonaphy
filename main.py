from src.phylo import compress_alignment
import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
import random
import os
import numpy as np

from src.vi import DodonaphyVI as vi
from src.mcmc import DodonaphyMCMC as mcmc


def main():

    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
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

    # save dna to nexus
    path_write = "./out"
    os.mkdir(path_write)  # os.makedirs(path_write, exist_ok=True)
    dest = path_write + "/dna.nex"
    dna.write_to_path(dest, "nexus")

    # Run Dodonaphy variational inference
    vi.run_tips(dim, S, partials[:], weights, dists, path_write, epochs=10000, k_samples=10, n_draws=200, **prior)

    # Run Dodoanphy MCMC
    mcmc.run(dim, partials[:], weights, dists, path_write, epochs=10000, step_scale=0.01, save_period=50, **prior)


if __name__ == "__main__":
    main()
