import argparse
import os
import random
import time

import dendropy
import numpy as np
from dendropy.interop import raxml
from dendropy.model.birthdeath import birth_death_likelihood
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim

from src.phylo import compress_alignment


def run(args):
    if args.root_ext != "":
        args.root_ext = "_" + args.root_ext
    root_dir = os.path.abspath(os.path.join("./data", f"T{args.taxa}{args.root_ext}"))
    path_write = get_path(root_dir, args)

    prior = {"birth_rate": args.birth, "death_rate": args.death}
    dna, simtree = get_dna(root_dir, prior)
    partials, weights = compress_alignment(dna)

    if args.start == "RAxML":
        rx = raxml.RaxmlRunner()
        tree_init = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    else:
        tree_init = simtree
    dists = tip_distances(tree_init, args.taxa)
    save_period = max(int(args.epoch / args.draws), 1)

    start = time.time()
    if args.infer == "mcmc":
        from src.mcmc import DodonaphyMCMC as mcmc

        mcmc.run(
            args.dim,
            partials[:],
            weights,
            dists,
            path_write,
            epochs=args.epoch,
            step_scale=args.step,
            save_period=save_period,
            n_grids=args.grids,
            n_trials=args.trials,
            max_scale=args.max_scale,
            nChains=args.chains,
            burnin=args.burn,
            connect_method=args.connect,
            embed_method=args.embed,
            curvature=args.curv,
            **prior,
        )

    if args.infer == "vi":
        from src.vi import DodonaphyVI

        DodonaphyVI.run(
            args.dim,
            args.taxa,
            partials[:],
            weights,
            dists,
            path_write,
            epochs=args.epoch,
            k_samples=args.importance,
            n_draws=args.draws,
            n_grids=args.grids,
            n_trials=args.strials,
            max_scale=args.max_scale,
            lr=args.learn,
            embed_method=args.embed,
            connect_method=args.embed,
            curvature=args.curv,
            **prior,
        )
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    print(f"Time taken for {args.taxa} taxa with {args.epoch} epochs: {m}m {s}s")


def get_path(root_dir, args):
    if args.doSave == False:
        return None

    if args.exp_ext != "":
        args.exp_ext = "_" + args.exp_ext
    exp_method = "%s_%s" % (args.embed, args.connect)

    if args.infer == "vi":
        if args.doSave:
            lnLr = -int(np.log10(args.learn))
            method_dir = os.path.join(root_dir, "vi", exp_method)
            path_write = os.path.join(
                method_dir,
                "d%i_lr%i_k%i%s" % (args.dim, lnLr, args.importance, args.exp_ext),
            )
            print(f"Saving to {path_write}")
        else:
            path_write = None

    elif args.infer == "mcmc":
        if args.doSave:
            method_dir = os.path.join(root_dir, "mcmc", exp_method)
            path_write = os.path.join(
                method_dir, "d%d_c%d%s" % (args.dim, args.chains, args.exp_ext)
            )
            print(f"Saving to {path_write}")
        else:
            path_write = None

    if path_write is not None:
        if not os.path.exists(method_dir):
            os.makedirs(method_dir, exist_ok=False)
        os.mkdir(path_write)
    return path_write


def get_dna(root_dir, prior):
    tree_path = os.path.join(root_dir, "simtree.nex")
    tree_info_path = os.path.join(root_dir, "simtree.info")
    dna_path = os.path.join(root_dir, "dna.nex")
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
            birth_rate=prior["birth_rate"],
            death_rate=prior["death_rate"],
            num_extant_tips=args.taxa,
            rng=rng,
        )
        dna = simulate_discrete_chars(
            seq_len=args.seq,
            tree_model=simtree,
            seq_model=dendropy.model.discrete.Jc69(),
            rng=rng,
        )

        simtree.write(path=tree_path, schema="nexus")
        dna.write_to_path(dest=dna_path, schema="nexus")
        LL = birth_death_likelihood(
            tree=simtree, birth_rate=prior["birth_rate"], death_rate=prior["death_rate"]
        )
        with open(tree_info_path, "w") as f:
            f.write("Log Likelihood: %f\n" % LL)
            simtree.write_ascii_p
    return dna, simtree


def tip_distances(tree0, n_taxa):
    """Get tip pair-wise tip distances"""
    dists = np.zeros((n_taxa, n_taxa))
    pdc = tree0.phylogenetic_distance_matrix()
    for i, t1 in enumerate(tree0.taxon_namespace[:-1]):
        for j, t2 in enumerate(tree0.taxon_namespace[i + 1 :]):
            dists[i][i + j + 1] = pdc(t1, t2)
    return dists + dists.transpose()


def init_parser():
    parser = argparse.ArgumentParser(
        prog="Dodonaphy",
        description="Compute a Bayesian phylogenetic posterior from a hyperbolic embedding.",
    )
    parser.add_argument("--dim", "-D", default=5, type=int, help="Embedding dimensions")
    parser.add_argument("--taxa", "-S", default=17, type=int, help="Number of taxa.")
    parser.add_argument("--seq", "-L", default=100, type=int, help="Sequence length.")
    parser.add_argument(
        "--epoch", "-n", default=1000, type=int, help="Epochs (iterations)."
    )
    parser.add_argument(
        "--draws",
        "-d",
        default=1000,
        type=int,
        help="Number of samples to draw from distribution.",
    )
    parser.add_argument(
        "--connect",
        "-C",
        default="nj",
        choices=("nj", "mst", "geodesics", "incentre", "mst_choice"),
        help="Connection method to form a tree from embedded points.",
    )
    parser.add_argument(
        "--embed",
        "-e",
        default="simple",
        choices=("simple", "wrap"),
        help="Embedded method from Euclidean to Hyperbolic space.",
    )
    parser.add_argument(
        "--doSave",
        "-s",
        default=True,
        type=bool,
        help="Whether to save the simulation.",
    )
    parser.add_argument(
        "--infer",
        "-i",
        default="mcmc",
        choices=("mcmc", "vi"),
        help="Inference method: MCMC or Variational Inference.",
    )
    parser.add_argument(
        "--curv", "-c", default=-1.0, type=float, help="Hyperbolic curvature."
    )
    parser.add_argument(
        "--start",
        "-t",
        default="true",
        choices=("true", "RAxML"),
        help="Starting tree to embed.",
    )
    parser.add_argument(
        "--root_ext",
        default="",
        type=str,
        help="Add a suffix to the root directory data/T[S]_[root_ext]",
    )
    parser.add_argument(
        "--exp_ext",
        default="",
        type=str,
        help="Add a suffix to the experimental directory data/T[S]/*/d*_c*_crv*_[exp_ext]",
    )

    # VI parameters
    parser.add_argument(
        "--importance",
        "-k",
        default=1,
        type=int,
        help="Number of tree samples for each epoch in Variational inference.",
    )
    parser.add_argument(
        "--learn", "-r", default=1e-1, type=float, help="Learning rate."
    )

    # MCMC parameters
    parser.add_argument(
        "--step", "-x", default=0.001, type=float, help="Initial step scale for MCMC."
    )
    parser.add_argument(
        "--chains", "-N", default=5, type=int, help="Number of MCMC chains."
    )
    parser.add_argument(
        "--burn", "-b", default=0, type=int, help="Number of burn in iterations."
    )

    # MST parameters
    parser.add_argument(
        "--trials",
        default=10,
        type=int,
        help="Number of initial embeddings to select from per grid for mst.",
    )
    parser.add_argument(
        "--grids",
        default=10,
        type=int,
        help="Number grid scales for selecting inital embedding for mst.",
    )
    parser.add_argument(
        "--max_scale",
        default=1,
        type=float,
        help="Maximum radius for mst internal node positions relative to minimum leaf radius.",
    )
    parser.add_argument(
        "--birth", default=2.0, type=float, help="Birth rate of simulated tree."
    )
    parser.add_argument(
        "--death", default=0.5, type=float, help="Death rate of simulated tree."
    )
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    run(args)
