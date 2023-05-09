import argparse


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
        choices=("mcmc", "vi", "hmap", "hlaplace", "simulate", "brute"),
        help="Inf: Inference method for Bayesian inference:\
        [mcmc]: MCMC\
        [vi]: Variational bayesian inference.\
        Use [hmap] to maximise the posterior of the hyperbolic embedding.\
        Use [hlaplace] to maximise the posterior of the embedding (hmap) and\
        then draw samples from a laplace approximation around the map.\
        Use [simulate] to simulate dna from a birth death tree.\
        Use [brute] to perform a grid search of the first node location.",
    )
    parser.add_argument(
        "--prior",
        default="None",
        choices=("None", "gammadir", "birthdeath", "normal", "uniform"),
        help=(
            "Inf: Which prior to use: no prior, Gamma-Dirichlet, Birth-Death\
         , normal or uniform embedding location."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        default="JC69",
        choices=("JC69", "GTR"),
        help="Inf: Markov model of nucleotide of evolution."

    )
    parser.add_argument(
        "--connect",
        "-C",
        default="nj",
        choices=("nj", "geodesics", "fix", "nj-r"),
        help="Inf: Connection method to form a tree from embedded points.\
        Fixing the topology is experimental and requires inputting a --start tree. \
        nj-r: Dodonaphy can use decenttree's implementation of to RapidNJ. First\
        pydecenttree must be installed from https://github.com/iqtree/decenttree.",
    )
    parser.add_argument(
        "--epochs", "-n", default=1000, type=int, help="Number of iterations/ epochs."
    )
    parser.add_argument(
        "--use_bito",
        dest="use_bito",
        action="store_true",
        help="Inf: Use bito (calling BEAGLE) to calculate likelihoods. \
            Bito reads in a fasta file with the same name as --path_dna.\
            E.g. both msa.fasta and msa.nex must be present, but only give\
            msa.nex to --path_dna.",
    )
    parser.set_defaults(use_bito=False)

    # i/o
    parser.add_argument(
        "--start",
        "-t",
        default="NJ",
        help="I/O: Path to starting tree in nexus format. Path is relative to\
        path_root. If set to RAxML, a RAxML tree will be found and used. If\
        set to 'Random, random exponentially distributed distances will be\
        used. If set to 'NJ' the neighbour joining tree will be formed from\
        the JC adjusted hamming distances, then the patristic distances on\
        this tree are used.",
    )
    parser.add_argument(
        "--path_root",
        default="",
        type=str,
        help="I/O: Specify the root directory, which should contain a nexus file.\
        If empty uses default ./data/T[taxa]",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="I/O: Add a suffix to the experimental directory path_root/name_[suffix]",
    )
    parser.add_argument(
        "--path_dna",
        default="dna.nex",
        type=str,
        help="I/O: File name of dna nexus file. Path is realtive to path_root.",
    )
    parser.add_argument(
        "--no-save",
        dest="no_save",
        action="store_true",
        help="I/O: Dry run, not saving to file.",
    )
    parser.set_defaults(no_save=False)
    parser.add_argument(
        "--write-dists",
        dest="write_dists",
        action="store_true",
        help="I/O: In MCMC mode, save proposal distances. In hmap mode, save the embedding locations at each step.",
    )
    parser.set_defaults(write_dists=False)

    # embedding
    parser.add_argument("--dim", "-D", default=3, type=int, help="Embedding dimensions")
    parser.add_argument(
        "--embed",
        "-e",
        default="up",
        choices=("up", "wrap"),
        help="Embed: Embedded method from Euclidean to Hyperbolic space.",
    )
    parser.add_argument(
        "--curv", "-c", default=-1.0, type=float, help="Embed: Hyperbolic curvature. For VI and HMAP modes, this is optimised from the given value."
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
        "--circle_leaf",
        dest="normalise_leaves",
        action="store_true",
        help="Embed: Whether to normalise the leaves to a single raduis. Currently\
        only implemented in MCMC and HMAP.",
    )
    parser.set_defaults(normalise_leaves=False)
    parser.add_argument(
        "--matsumoto",
        dest="matsumoto",
        action="store_true",
        help="Embed: Apply the Matsumoto et al 2020 distance adjustment. NB: hydra+\
            does not account for this which could lead to a bad initial\
            embedding. Only implemented in MCMC and HMAP.",
    )
    parser.set_defaults(matsumoto=False)

    # MCMC parameters
    parser.add_argument(
        "--step",
        "-x",
        default=0.001,
        type=float,
        help="MCMC: Initial step scale for MCMC.",
    )
    parser.add_argument(
        "--chains", "-N", default=4, type=int, help="MCMC: Number of MCMC chains."
    )
    parser.add_argument(
        "--burn", "-b", default=0, type=int, help="MCMC: Number of burn in iterations."
    )
    parser.add_argument(
        "--loss_fn",
        default="likelihood",
        choices=("likelihood", "pair_likelihood", "hypHC", "none"),
        help="MCMC: Loss function for MCMC and MAP. Not implemented in VI.\
        Use none for no model, prior only.",
    )
    parser.add_argument(
        "--swap_period",
        default=100,
        type=int,
        help="MCMC: Number MCMC generations before considering swapping chains.",
    )
    parser.add_argument(
        "--n_swaps",
        default=5,
        type=int,
        help="MCMC: Number of MCMC chain swap moves considered every swap_period.",
    )
    parser.add_argument(
        "--warm_up",
        default=1000,
        type=int,
        help="MCMC: Number of iterations before using adaptive covariance.\
            Tune a single step for all tips before this iteration.",
    )
    parser.add_argument(
        "--mcmc_alg",
        default="RAM",
        choices=("RAM", "tune", "AM"),
        help="MCMC: algorithm used. Robust adaptive Metropolis (RAM) is only\
            used after warm_up. Tune: the covariance by multipying by a\
            constant to achieve target acceptance rate (used during warm_up).\
            Adaptive Metropolis (AM) won't tune acceptance rate.",
    )

    # tuning parameters
    parser.add_argument(
        "--temp",
        default=None,
        type=float,
        help="TUNE: Temperature for soft neighbour joining. Towards 0 is\
            'colder', which increases accuracy, but reduces gradient\
            information.",
    )
    parser.add_argument(
        "--learn",
        "-r",
        default=1e-1,
        type=float,
        help="TUNE: Initial learning rate. Also for learning MCMC steps.",
    )

    # VI parameters
    parser.add_argument(
        "--draws",
        "-d",
        default=1000,
        type=int,
        help="VI/MCMC: Number of samples to draw from distribution in hlaplace, VI, and MCMC (via thinning).",
    )
    parser.add_argument(
        "--importance",
        "-k",
        default=1,
        type=int,
        help="VI: Number of tree samples for each epoch in Variational inference.",
    )
    parser.add_argument(
        "--boosts",
        default=1,
        type=int,
        help="VI: Total number of mixtures to boost variational distribution.",
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


def validate(args):
    if args.infer in ("vi", "ml", "map", "hmap", "hlaplace"):
        if args.temp is None:
            raise ValueError("--temp not be None for this mode of inference.")
        if args.temp <= 0.0:
            raise ValueError("--temp must be greater than 0 for this mode of inference.")
    if args.connect == "fix":
        if args.start is None:
            raise ValueError("--start tree cannot be None when the topology is fixed (--connect fix).")
    if args.use_bito:
        if args.start == "NJ":
            raise ValueError("--start tree must be provided to in order to --use_bito.\
                             Just need to implement init_bito better to cope without.")
