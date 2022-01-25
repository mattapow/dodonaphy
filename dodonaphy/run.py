"""Run Dodonaphy module"""
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

from dodonaphy import utils
from dodonaphy.mcmc import DodonaphyMCMC as mcmc
from dodonaphy.ml import ML
from dodonaphy.phylo import compress_alignment
from dodonaphy.Cphylo import compress_alignment_np
from dodonaphy.vi import DodonaphyVI


def run(args):
    """Run dodonaphy.
    Using an embedding for MCMC, or embedding variational inference,
    or using a distance matrix for maximum likelihood."""
    if args.path_root == "":
        root_dir = os.path.abspath(os.path.join("./data", f"T{args.taxa}"))
    else:
        root_dir = os.path.abspath(args.path_root)

    if args.infer == "simulate":
        simulate_tree(root_dir, args.birth, args.death, args.taxa, args.seq)
        return

    path_write = get_path(root_dir, args)
    dna = read_dna(root_dir, args.dna_path)

    if args.start == "RAxML":
        print("Finding RAxML tree.")
        rax = raxml.RaxmlRunner()
        start_tree = rax.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
        tree_path = os.path.join(root_dir, "start_tree.nex")
        start_tree.write(path=tree_path, schema="nexus")
    else:
        start_tree = read_tree(root_dir, file_name=args.start)

    dists_phylo = utils.tip_distances(start_tree, args.taxa)
    dists_matsumoto = np.arccosh(np.exp(dists_phylo))
    save_period = max(int(args.epochs / args.draws), 1)

    start = time.time()
    if args.infer == "mcmc":
        partials, weights = compress_alignment_np(dna)
        mcmc.run(
            args.dim,
            partials[:],
            weights,
            dists_matsumoto,
            path_write,
            epochs=args.epochs,
            step_scale=args.step,
            save_period=save_period,
            n_grids=args.grids,
            n_trials=args.trials,
            max_scale=args.max_scale,
            n_chains=args.chains,
            burnin=args.burn,
            connector=args.connect,
            embedder=args.embed,
            curvature=args.curv,
            normalise_leaf=args.normalise_leaves,
            loss_fn=args.loss_fn,
        )
    elif args.infer == "vi":
        partials, weights = compress_alignment(dna)
        DodonaphyVI.run(
            args.dim,
            args.taxa,
            partials[:],
            weights,
            dists_matsumoto,
            path_write,
            epochs=args.epochs,
            k_samples=args.importance,
            n_draws=args.draws,
            n_grids=args.grids,
            n_trials=args.trials,
            max_scale=args.max_scale,
            lr=args.learn,
            embedder=args.embed,
            connector=args.connect,
            curvature=args.curv,
            soft_temp=args.temp,
        )
    elif args.infer == "ml":
        partials, weights = compress_alignment(dna)
        mymod = ML(
            partials[:],
            weights,
            dists=dists_matsumoto,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
        )
        mymod.learn(epochs=args.epochs, learn_rate=args.learn, path_write=path_write)

    mins, secs = divmod(time.time() - start, 60)
    hrs, mins = divmod(mins, 60)
    print(
        f"Time taken for {args.taxa} taxa with {args.epochs} epochs: {int(hrs)}:{int(mins)}:{int(secs)}\n"
    )


def get_path(root_dir, args):
    """Generate and return experiment path"""
    if args.doSave is False:
        return None

    if args.exp_ext != "":
        args.exp_ext = "_" + args.exp_ext
    exp_method = f"{args.embed}_{args.connect}"

    if args.infer == "vi":
        ln_lr = -int(np.log10(args.learn))
        method_dir = os.path.join(root_dir, "vi", exp_method)
        path_write = os.path.join(
            method_dir,
            f"d{args.dim}_lr{ln_lr}_k{args.importance}{args.exp_ext}",
        )
        print(f"Saving to {path_write}")

    elif args.infer == "mcmc":
        method_dir = os.path.join(root_dir, "mcmc", exp_method)
        path_write = os.path.join(
            method_dir, f"d{args.dim}_c{args.chains}{args.exp_ext}"
        )
        print(f"Saving to {path_write}")

    elif args.infer == "ml":
        assert (
            args.connect == "nj"
        ), "Maximum likelihood only works on neighbour joining. This is since\n\
            it it the only purely distance-based connection method\n\
            implemented. Other methods depend on embedding locations."
        ln_rate = -int(np.log10(args.learn))
        ln_tau = -int(np.log10(args.temp))
        method_dir = os.path.join(root_dir, "ml", args.connect)
        path_write = os.path.join(method_dir, f"lr{ln_rate}_tau{ln_tau}{args.exp_ext}")
        print(f"Saving to {path_write}")

    if path_write is not None:
        if not os.path.exists(method_dir):
            try:
                os.makedirs(method_dir, exist_ok=False)
            except OSError:
                print(
                    f"Failed making directoy {method_dir}. Possibly an array job on HPC."
                )
        os.mkdir(path_write)
    return path_write


def simulate_tree(root_dir, birth_rate, death_rate, n_taxa, seq_len):
    """Simulate a birth death tree and save it.

    Args:
        root_dir ([type]): [description]
        birth_rate ([type]): [description]
        death_rate ([type]): [description]
        n_taxa ([type]): [description]
        seq_len ([type]): [description]
    """
    os.makedirs(root_dir, exist_ok=False)
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(
        birth_rate=birth_rate,
        death_rate=death_rate,
        num_extant_tips=n_taxa,
        rng=rng,
    )
    dna = simulate_discrete_chars(
        seq_len=seq_len,
        tree_model=simtree,
        seq_model=dendropy.model.discrete.Jc69(),
        rng=rng,
    )

    tree_info_path = os.path.join(root_dir, "start_tree.info")
    tree_path = os.path.join(root_dir, "start_tree.nex")
    dna_path = os.path.join(root_dir, "dna.nex")

    simtree.write(path=tree_path, schema="nexus")
    dna.write_to_path(dest=dna_path, schema="nexus")
    ln_like = birth_death_likelihood(
        tree=simtree, birth_rate=birth_rate, death_rate=death_rate
    )
    with open(tree_info_path, "w", encoding="utf_8") as file:
        file.write(f"Log Likelihood: {ln_like}\n")
        simtree.write_ascii_plot(file)


def read_tree(root_dir, file_name="start_tree.nex"):
    """Read a saved nexus tree using dendropy."""
    tree_path = os.path.join(root_dir, file_name)
    return dendropy.Tree.get(path=tree_path, schema="nexus")


def read_dna(root_dir, file_name="dna.nex"):
    """Get dna from a saved simulated tree."""
    dna_path = os.path.join(root_dir, file_name)
    dna = dendropy.DnaCharacterMatrix.get(path=dna_path, schema="nexus")
    return dna


def init_parser():
    """Initialise argument parser."""
    parser = argparse.ArgumentParser(
        prog="Dodonaphy",
        description="Compute a Bayesian phylogenetic posterior from a hyperbolic embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dim", "-D", default=5, type=int, help="Embedding dimensions")
    parser.add_argument("--taxa", "-S", type=int, required=True, help="Number of taxa.")
    parser.add_argument("--seq", "-L", type=int, help="Sequence length.")
    parser.add_argument(
        "--epochs", "-n", default=1000, type=int, help="Epochs (iterations)."
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
        choices=("nj", "geodesics"),
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
        "--infer",
        "-i",
        default="mcmc",
        choices=("mcmc", "vi", "ml", "simulate"),
        help="Inference method: MCMC or Variational Inference for Bayesian\
        inference. Use ml to maximise the likelihod of a similarity matrix.\
        Use [simulate] to simulate dna from a birth death tree.",
    )
    parser.add_argument(
        "--curv", "-c", default=-1.0, type=float, help="Hyperbolic curvature."
    )
    parser.add_argument(
        "--start",
        "-t",
        default="data/start_tree.nex",
        help="Path to starting tree in nexus format. If set to RAxML, a RAxML\
        tree will be found and used.",
    )
    parser.add_argument(
        "--normalise_leaf",
        dest="normalise_leaves",
        action="store_true",
        help="Whether to normalise the leaves to a single raduis. Currently\
        only implemented in MCMC.",
    )
    parser.add_argument(
        "--free_leaf",
        dest="normalise_leaves",
        action="store_false",
        help="Whether to normalise the leaves to a single raduis. Currently\
        only implemented in MCMC.",
    )
    parser.set_defaults(normalise_leaves=False)

    # i/o
    parser.add_argument(
        "--path_root",
        default="",
        type=str,
        help="Specify the root directory, which should contain a nexus file.\
        If empty uses default ./data/T[taxa]",
    )
    parser.add_argument(
        "--exp_ext",
        default="",
        type=str,
        help="Add a suffix to the experimental directory path_root/*/d*_c*_[exp_ext]",
    )
    parser.add_argument(
        "--dna_path",
        default="dna.nex",
        type=str,
        help="File name of dna nexus file in contained in root directory.",
    )
    parser.add_argument(
        "--save",
        dest="doSave",
        action="store_true",
        help="Whether to save the simulation.",
    )
    parser.add_argument(
        "--no-save",
        dest="doSave",
        action="store_false",
        help="Whether to save the simulation.",
    )
    parser.set_defaults(doSave=True)

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
        "--step", "-x", default=0.1, type=float, help="Initial step scale for MCMC."
    )
    parser.add_argument(
        "--chains", "-N", default=5, type=int, help="Number of MCMC chains."
    )
    parser.add_argument(
        "--burn", "-b", default=0, type=int, help="Number of burn in iterations."
    )
    parser.add_argument(
        "--loss_fn",
        default="likelihood",
        choices=("likelihood", "pair_likelihood", "hypHC"),
        help="Loss function for MCMC and ML. Not implemented in VI.",
    )
    parser.add_argument(
        type=int,
    )
    parser.add_argument(
        default=10,
        type=int,
    )

    # Tree simulation parameters
    parser.add_argument(
        "--birth", default=2.0, type=float, help="Birth rate of simulated tree."
    )
    parser.add_argument(
        "--death", default=0.5, type=float, help="Death rate of simulated tree."
    )

    # "soft" parameters
    parser.add_argument(
        "--temp",
        default=None,
        type=float,
        help="Temperature for soft neighbour joining",
    )
    return parser


def main():
    """Main entry point"""
    parser = init_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
