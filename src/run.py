import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.model.birthdeath import birth_death_likelihood
from dendropy.interop import raxml
import random
import os
import numpy as np
import time
import sys
from src.phylo import compress_alignment


def main():
    if len(sys.argv) != 2:
        print("Usage: %s value" % sys.argv[0])
        sys.exit()
    try:
        input_dim = float(sys.argv[1])
    except Exception:
        print("Argument not a float.")
        sys.exit()

    # Simulation parameters
    dim = int(input_dim)  # number of dimensions for embedding
    S = 17  # number of sequences to simulate
    L = 1000  # length of sequences to simulate
    prior = {"birth_rate": 2., "death_rate": .5}
    epochs = 500000      # number of epochs
    n_draws = 1000      # number of trees drawn from final distribution
    connect_method = 'nj'      # 'incentre', 'mst', 'geodesics', 'nj', 'mst_choice'
    embed_method = 'simple'    # 'simple' or 'wrap'
    doSave = True
    inference = 'mcmc'
    curvature = -1.
    start_tree = 'true_tree'       # 'RAxML' or 'true_tree'

    # VI parameters
    k_samples = 1       # tree samples per elbo calculation
    lr = 1e-1

    # MCMC parameters
    step_scale = .001
    save_period = max(int(epochs/n_draws), 1)
    nChains = 5
    burnin = 0

    # MST parameters
    n_trials = 10      # number of initial embeddings to select from per grid for mst
    n_grids = 10       # number grid scales for selecting inital embedding for mst
    max_scale = 1       # for mst internal node positions

    # Experiment folder
    root_dir = os.path.abspath(os.path.join("../data", "T%d" % (S)))
    tree_path = os.path.join(root_dir, "simtree.nex")
    tree_info_path = os.path.join(root_dir, "simtree.info")
    dna_path = os.path.join(root_dir, "dna.nex")
    exp_method = "%s_%s" % (embed_method, connect_method)

    if inference == 'vi':
        if doSave:
            lnLr = -int(np.log10(lr))
            method_dir = os.path.join(root_dir, "vi", exp_method)
            path_write = os.path.join(
                method_dir, "d%i_lr%i_k%i_crv%d" %
                (dim, lnLr, k_samples, -curvature*100))
            print(f"Saving to {path_write}")
        else:
            path_write = None

    elif inference == 'mcmc':
        if doSave:
            method_dir = os.path.join(root_dir, "mcmc", exp_method)
            path_write = os.path.join(
                method_dir, "d%d_c%d_crv%d" %
                (dim, nChains, -curvature*100))
            print(f"Saving to {path_write}")
        else:
            path_write = None

    try:
        # Try loading in the simTree and dna
        simtree = dendropy.Tree.get(path=tree_path, schema="nexus")
        dna = dendropy.DnaCharacterMatrix.get(path=dna_path, schema="nexus")
    except (FileExistsError, FileNotFoundError):
        # Make experiment folder
        os.makedirs(root_dir, exist_ok=False)

        # simulate a tree
        rng = random.Random(1)
        simtree = treesim.birth_death_tree(
            birth_rate=prior['birth_rate'], death_rate=prior['death_rate'], num_extant_tips=S, rng=rng)
        dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

        # save simtree
        simtree.write(path=tree_path, schema="nexus")

        # save dna to nexus
        dna.write_to_path(dest=dna_path, schema="nexus")

        # save simTree info log-likelihood
        LL = birth_death_likelihood(tree=simtree, birth_rate=prior['birth_rate'], death_rate=prior['death_rate'])
        with open(tree_info_path, 'w') as f:
            f.write('Log Likelihood: %f\n' % LL)
            simtree.write_ascii_plot(f)

    partials, weights = compress_alignment(dna)

    if start_tree == 'RAxML':
        rx = raxml.RaxmlRunner()
        tree0 = rx.estimate_tree(
            char_matrix=dna,
            raxml_args=["--no-bfgs"])
    else:
        tree0 = simtree

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = tree0.phylogenetic_distance_matrix()
    for i, t1 in enumerate(tree0.taxon_namespace[:-1]):
        for j, t2 in enumerate(tree0.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    if path_write is not None:
        if not os.path.exists(method_dir):
            os.makedirs(method_dir, exist_ok=False)
        os.mkdir(path_write)

    start = time.time()
    if inference == 'mcmc':
        # Run Dodoanphy MCMC
        from src.mcmc import DodonaphyMCMC as mcmc
        mcmc.run(dim, partials[:], weights, dists, path_write,
                 epochs=epochs, step_scale=step_scale, save_period=save_period,
                 n_grids=n_grids, n_trials=n_trials, max_scale=max_scale, nChains=nChains,
                 burnin=burnin, connect_method=connect_method, embed_method=embed_method,
                 curvature=curvature, **prior)

    if inference == 'vi':
        # Run Dodonaphy variational inference
        from src.vi import DodonaphyVI
        DodonaphyVI.run(dim, S, partials[:], weights, dists, path_write,
                        epochs=epochs, k_samples=k_samples, n_draws=n_draws,
                        n_grids=n_grids, n_trials=n_trials,
                        max_scale=max_scale, lr=lr, embed_method=embed_method,
                        connect_method=connect_method, curvature=curvature, **prior)
    end = time.time()
    seconds = end-start
    m, s = divmod(seconds, 60)
    print("Time taken for %d taxa: %dm %ds" % (S, m, s))


if __name__ == "__main__":
    main()
