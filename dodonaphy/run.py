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
from dodonaphy.map import MAP
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import compress_alignment, calculate_pairwise_distance
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
    partials, weights = compress_alignment(dna)

    if args.start == "None":
        print("Computing distances from sequences:", end="", flush=True)
        dists = calculate_pairwise_distance(dna, adjust=None)
        print(" done.", flush=True)
        tip_labels = dna.taxon_namespace.labels()
    elif args.start == "RAxML":
        print("Finding RAxML tree.")
        rax = raxml.RaxmlRunner()
        start_tree = rax.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
        tree_path = os.path.join(root_dir, "start_tree.nex")
        start_tree.write(path=tree_path, schema="nexus")
        dists = utils.tip_distances(start_tree, args.taxa)
        tip_labels = start_tree.taxon_namespace.labels()
    else:
        start_tree = read_tree(root_dir, file_name=args.start)
        args.taxa = len(start_tree)
        dists = utils.tip_distances(start_tree, args.taxa)
        tip_labels = start_tree.taxon_namespace.labels()

    if args.matsumoto:
        dists = np.arccosh(np.exp(dists))
    save_period = max(int(args.epochs / args.draws), 1)

    start = time.time()
    if args.infer == "mcmc":
        mcmc.run(
            args.dim,
            partials[:],
            weights,
            dists,
            path_write,
            epochs=args.epochs,
            step_scale=args.step,
            save_period=save_period,
            n_chains=args.chains,
            burnin=args.burn,
            connector=args.connect,
            embedder=args.embed,
            curvature=args.curv,
            normalise_leaf=args.normalise_leaves,
            loss_fn=args.loss_fn,
            swap_period=args.swap_period,
            n_swaps=args.n_swaps,
            matsumoto=args.matsumoto,
            tip_labels=tip_labels,
        )
    elif args.infer == "vi":
        DodonaphyVI.run(
            args.dim,
            args.taxa,
            partials[:],
            weights,
            dists,
            path_write,
            epochs=args.epochs,
            k_samples=args.importance,
            n_draws=args.draws,
            lr=args.learn,
            embedder=args.embed,
            connector=args.connect,
            curvature=args.curv,
            soft_temp=args.temp,
            tip_labels=tip_labels,
        )
    elif args.infer == "ml":
        assert args.temp > 0.0, "Temperature must be greater than 0."
        partials, weights = compress_alignment(dna)
        mymod = MAP(
            partials[:],
            weights,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            prior="None",
            tip_labels=tip_labels,
        )
        mymod.learn(epochs=args.epochs, learn_rate=args.learn, path_write=path_write)

    elif args.infer == "map":
        assert args.temp > 0.0, "Temperature must be greater than 0."
        partials, weights = compress_alignment(dna)
        mymod = MAP(
            partials[:],
            weights,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            prior=args.prior,
            tip_labels=tip_labels,
        )
        mymod.learn(epochs=args.epochs, learn_rate=args.learn, path_write=path_write)
    
    elif args.infer =="hmap":
        assert args.temp > 0.0, "Temperature must be greater than 0."
        partials, weights = compress_alignment(dna)
        mymod = HMAP(
            partials[:],
            weights,
            dim = args.dim,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            prior=args.prior,
            tip_labels=tip_labels,
            matsumoto=args.matsumoto,
        )
        mymod.learn(epochs=args.epochs, learn_rate=args.learn, path_write=path_write)


    mins, secs = divmod(time.time() - start, 60)
    hrs, mins = divmod(mins, 60)
    print(
        f"Time taken for {args.taxa} taxa with {args.epochs} epochs: {int(hrs)}:{int(mins)}:{int(secs)}\n"
    )


def get_path(root_dir, args):
    """Generate and return experiment path"""
    if args.no_save is True:
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

    elif args.infer == "mcmc":
        method_dir = os.path.join(root_dir, "mcmc", exp_method)
        path_write = os.path.join(
            method_dir, f"d{args.dim}_c{args.chains}{args.exp_ext}"
        )

    elif args.infer in ("ml", "map", "hmap"):
        ln_rate = -int(np.log10(args.learn))
        ln_tau = -int(np.log10(args.temp))
        method_dir = os.path.join(root_dir, args.infer, args.connect)
        path_write = os.path.join(method_dir, f"lr{ln_rate}_tau{ln_tau}{args.exp_ext}")

    if path_write is not None:
        print(f"Saving to {path_write}")
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
    # inference
    parser.add_argument(
        "--infer",
        "-i",
        default="mcmc",
        choices=("mcmc", "vi", "ml", "map", "hmap", "simulate"),
        help="Inf: Inference method: MCMC or Variational Inference for Bayesian\
        inference. Use map to maximise the posterior of a similarity matrix.\
        Use hmap to maximise the posterior of the embedding. Use [simulate] to\
        simulate dna from a birth death tree.",
    )
    parser.add_argument(
        "--prior",
        default="None",
        choices=("None", "gammadir", "birthdeath"),
        help=("Inf: Which prior to use: no prior, Gamma-Dirichlet or Birth-Death."),
    )
    parser.add_argument(
        "--connect",
        "-C",
        default="nj",
        choices=("nj", "geodesics"),
        help="Inf: Connection method to form a tree from embedded points.",
    )

    # i/o
    parser.add_argument(
        "--start",
        "-t",
        default="None",
        help="I/O: Path to starting tree in nexus format. If set to RAxML, a RAxML\
        tree will be found and used. If set to 'None' the distances will be\
        inferred from the sequences.",
    )
    parser.add_argument(
        "--path_root",
        default="",
        type=str,
        help="I/O: Specify the root directory, which should contain a nexus file.\
        If empty uses default ./data/T[taxa]",
    )
    parser.add_argument(
        "--exp_ext",
        default="",
        type=str,
        help="I/O: Add a suffix to the experimental directory path_root/*/d*_c*_[exp_ext]",
    )
    parser.add_argument(
        "--dna_path",
        default="dna.nex",
        type=str,
        help="I/O: File name of dna nexus file in contained in root directory.",
    )
    parser.add_argument(
        "--no-save",
        dest="no_save",
        action="store_true",
        help="I/O: Dry run, not saving to file.",
    )
    parser.set_defaults(no_save=False)

    # embedding
    parser.add_argument("--dim", "-D", default=5, type=int, help="Embedding dimensions")
    parser.add_argument(
        "--embed",
        "-e",
        default="up",
        choices=("up", "wrap"),
        help="Embed: Embedded method from Euclidean to Hyperbolic space.",
    )
    parser.add_argument(
        "--curv", "-c", default=-1.0, type=float, help="Embed: Hyperbolic curvature."
    )
    parser.add_argument(
        "--normalise_leaf",
        dest="normalise_leaves",
        action="store_true",
        help="Embed: Whether to normalise the leaves to a single raduis. NB: Hydra+\
            does not normalise leaves, which could lead to a bad initial\
            embedding. Currently only implemented in MCMC.",
    )
    parser.add_argument(
        "--free_leaf",
        dest="normalise_leaves",
        action="store_false",
        help="Embed: Whether to normalise the leaves to a single raduis. Currently\
        only implemented in MCMC.",
    )
    parser.set_defaults(normalise_leaves=False)
    parser.add_argument(
        "--matsumoto",
        dest="matsumoto",
        action="store_true",
        help="Embed: Apply the Matsumoto et al 2020 distance adjustment. NB: hydra+\
            does not account for this which could lead to a bad initial\
            embedding. Currently ony implemented in MCMC.",
    )
    parser.set_defaults(matsumoto=False)

    # MCMC parameters
    parser.add_argument(
        "--epochs", "-n", default=1000, type=int, help="MCMC: Iterations (VI epochs)."
    )
    parser.add_argument(
        "--step",
        "-x",
        default=0.1,
        type=float,
        help="MCMC: Initial step scale for MCMC.",
    )
    parser.add_argument(
        "--chains", "-N", default=5, type=int, help="MCMC: Number of MCMC chains."
    )
    parser.add_argument(
        "--burn", "-b", default=0, type=int, help="MCMC: Number of burn in iterations."
    )
    parser.add_argument(
        "--loss_fn",
        default="likelihood",
        choices=("likelihood", "pair_likelihood", "hypHC"),
        help="MCMC: Loss function for MCMC and MAP. Not implemented in VI.",
    )
    parser.add_argument(
        "--swap_period",
        default=1000,
        type=int,
        help="MCMC: Number MCMC generations before considering swapping chains.",
    )
    parser.add_argument(
        "--n_swaps",
        default=10,
        type=int,
        help="MCMC: Number of MCMC chain swap moves considered every swap_period.",
    )

    # tuning parameters
    parser.add_argument(
        "--temp",
        default=None,
        type=float,
        help="Tune: Temperature for soft neighbour joining. Towards 0 is\
            'colder', which increases accuracy, but reduces gradient\
            information.",
    )
    parser.add_argument(
        "--learn",
        "-r",
        default=1e-1,
        type=float,
        help="Initial learning rate. Also for learning MCMC steps.",
    )

    # VI parameters
    parser.add_argument(
        "--draws",
        "-d",
        default=1000,
        type=int,
        help="VI: Number of samples to draw from distribution.",
    )
    parser.add_argument(
        "--importance",
        "-k",
        default=1,
        type=int,
        help="VI: Number of tree samples for each epoch in Variational inference.",
    )

    # Tree simulation parameters
    parser.add_argument(
        "--taxa", "-S", type=int, help="Simu: Number of taxa to simulate."
    )
    parser.add_argument(
        "--seq", "-L", type=int, help="Simu: Sequence length for simulating a tree."
    )

    parser.add_argument(
        "--birth", default=2.0, type=float, help="Simu: Birth rate of simulated tree."
    )
    parser.add_argument(
        "--death", default=0.5, type=float, help="Simu: Death rate of simulated tree."
    )
    return parser


def main():
    """Main entry point"""
    parser = init_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
