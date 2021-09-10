import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.model.birthdeath import birth_death_likelihood
import random
import os
import numpy as np
import time
import sys

from dodonaphy.src.vi import DodonaphyVI
from dodonaphy.src.mcmc import DodonaphyMCMC as mcmc
from dodonaphy.src.phylo import compress_alignment


def main():
    if len(sys.argv) != 2:
        print("Usage: %s value" % sys.argv[0])
        sys.exit()
    try:
        input = float(sys.argv[1])
    except Exception:
        print("Argument not a float.")
        sys.exit()

    # Simulation parameters
    dim = int(input)  # number of dimensions for embedding
    S = 17  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}
    epochs = 10000      # number of epochs
    n_draws = 1000      # number of trees drawn from final distribution
    connect_method = 'nj'      # 'incentre', 'mst', 'geodesics', 'nj', 'mst_choice'
    embed_method = 'simple'     # 'simple' or 'wrap'
    doSave = True
    inference = 'vi'
    curvature = -1

    # VI parameters
    k_samples = 10       # tree samples per elbo calculation
    lr = 1e-3

    # MCMC parameters
    step_scale = .001
    save_period = max(int(epochs/n_draws), 1)
    nChains = 5
    burnin = 0

    # MST parameters
    n_trials = 100      # number of initial embeddings to select from per grid for mst
    n_grids = 100       # number grid scales for selecting inital embedding for mst
    max_scale = 1       # for mst internal node positions

    # Experiment folder
    path_write = "../data/T%d" % (S)
    treePath = "%s/simtree.nex" % path_write
    treeInfoPath = "%s/simtree.info" % path_write
    dnaPath = "%s/dna.nex" % path_write

    if inference == 'vi':
        if doSave:
            lnLr = -int(np.log10(lr))
            path_write_vi = os.path.abspath(os.path.join(
                path_write, "vi", "%s_%s_lr%i_k%i_d%i_crv%d" % (embed_method, connect_method, lnLr, k_samples, dim, curvature)))
        else:
            path_write_vi = None
        runVi = True
        runMcmc = False

    elif inference == 'mcmc':
        if doSave:
            path_write_mcmc = os.path.abspath(os.path.join(
                path_write, "mcmc", "%s_%s_c%d_d%d_crv%d" % (embed_method, connect_method, nChains, dim, curvature)))
        else:
            path_write_mcmc = None
        runMcmc = True
        runVi = False

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

    start = time.time()
    if runMcmc:
        # Run Dodoanphy MCMC
        if path_write_mcmc is not None:
            os.mkdir(path_write_mcmc)
        mcmc.run(dim, partials[:], weights, dists, path_write_mcmc,
                 epochs=epochs, step_scale=step_scale, save_period=save_period,
                 n_grids=n_grids, n_trials=n_trials, max_scale=max_scale, nChains=nChains,
                 burnin=burnin, connect_method=connect_method, embed_method=embed_method,
                 curvature=curvature, **prior)

    if runVi:
        # Run Dodonaphy variational inference
        if path_write_vi is not None:
            os.mkdir(path_write_vi)
        DodonaphyVI.run(dim, S, partials[:], weights, dists, path_write_vi,
                        epochs=epochs, k_samples=k_samples, n_draws=n_draws,
                        n_grids=n_grids, n_trials=n_trials,
                        max_scale=max_scale, lr=lr, embed_method=embed_method,
                        connect_method=connect_method, curvature=curvature, **prior)

    end = time.time()
    seconds = end-start
    m, s = divmod(seconds, 60)
    print("Time taken for %d taxa: %dm %ds" % (S, m, s))

    # Make folder for BEAST
    # path_write_beast = os.path.abspath(os.path.join(path_write, "beast"))
    # os.mkdir(path_write_beast)


if __name__ == "__main__":
    main()
