import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.model.birthdeath import birth_death_likelihood
import random
import os
import numpy as np

from src.vi import DodonaphyVI
from src.mcmc import DodonaphyMCMC as mcmc
from src.phylo import compress_alignment


def main():
    # Simulation parameters
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}
    epochs = 10000      # number of epochs
    n_draws = 1000      # number of trees drawn from final distribution
    init_trials = 1000   # number of initial embeddings to select from per grid
    init_grids = 100     # # number grid scales for selecting inital embedding
    max_scale = 1
    connect_method = 'mst'  # 'incentre', 'mst' or 'geodesics'

    # Experiment folder
    path_write = "../data/T%d_2" % (S)
    treePath = "%s/simtree.nex" % path_write
    treeInfoPath = "%s/simtree.info" % path_write
    dnaPath = "%s/dna.nex" % path_write

    # VI parameters
    k_samples = 10       # tree samples per elbo calculation
    lr = 1e-3
    embed_method = 'logit'  # TODO: wrapping method doesn't learn
    # path_write_vi = None
    path_write_vi = os.path.abspath(
        os.path.join(path_write, ("%s_%s_lr%i_k%i" % (embed_method, connect_method, -int(np.log10(lr)), k_samples))))
    runVi = True

    # MCMC parameters
    step_scale = .001
    save_period = max(int(epochs/n_draws), 1)
    nChains = 1
    burnin = 0
    path_write_mcmc = os.path.abspath(os.path.join(path_write, "mcmc_%s_step000001_1" % connect_method))
    runMcmc = False

    try:
        # Try loading in the simTree and dna
        simtree = dendropy.Tree.get(path=treePath, schema="nexus")
        dna = dendropy.DnaCharacterMatrix.get(path=dnaPath, schema="nexus")
    except (FileExistsError, FileNotFoundError):
        # Make experiment folder
        os.makedirs(path_write, exist_ok=False)

        # simulate a tree
        rng = random.Random(1)
        simtree = treesim.birth_death_tree(
            birth_rate=prior['birth_rate'], death_rate=prior['death_rate'], num_extant_tips=S, rng=rng)
        dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

        # save simtree
        simtree.write(path=treePath, schema="nexus")

        # save dna to nexus
        dna.write_to_path(dest=dnaPath, schema="nexus")

        # save simTree info log-likelihood
        LL = birth_death_likelihood(
            tree=simtree, birth_rate=prior['birth_rate'], death_rate=prior['death_rate'])
        with open(treeInfoPath, 'w') as f:
            f.write('Log Likelihood: %f\n' % LL)
            simtree.write_ascii_plot(f)

    partials, weights = compress_alignment(dna)

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    if runMcmc:
        # Run Dodoanphy MCMC
        os.mkdir(path_write_mcmc)
        mcmc.run(dim, partials[:], weights, dists, path_write_mcmc,
                 epochs=epochs, step_scale=step_scale, save_period=save_period,
                 init_grids=init_grids, init_trials=init_trials, nChains=nChains,
                 burnin=burnin, connect_method=connect_method, **prior)

    if runVi:
        # Run Dodonaphy variational inference
        if path_write_vi is not None:
            os.mkdir(path_write_vi)
        # path_write_vi = None
        DodonaphyVI.run(dim, S, partials[:], weights, dists, path_write_vi,
                        epochs=epochs, k_samples=k_samples, n_draws=n_draws,
                        init_grids=init_grids, init_trials=init_trials,
                        max_scale=max_scale, lr=lr, embed_method=embed_method,
                        connect_method=connect_method, **prior)

    # Make folder for BEAST
    # path_write_beast = os.path.abspath(os.path.join(path_write, "beast"))
    # os.mkdir(path_write_beast)


if __name__ == "__main__":
    main()
