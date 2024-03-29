"""Run Dodonaphy module"""
import json
import os
import random
import time
import warnings

import dendropy
import numpy as np
from dendropy.interop import raxml
from dendropy.model.birthdeath import birth_death_likelihood
from dendropy.model.discrete import simulate_discrete_chars
from dendropy.simulate import treesim

from dodonaphy import utils, cli, tree, Cpeeler
from dodonaphy.mcmc import DodonaphyMCMC as mcmc
from dodonaphy.hmap import HMAP
from dodonaphy.laplace import Laplace
from dodonaphy.phylo import (
    compress_alignment,
    calculate_pairwise_distance,
    compute_nucleotide_frequencies,
)
from dodonaphy.tensor_json import TensorDecoder
from dodonaphy.vi import DodonaphyVI
from dodonaphy.brute import Brute
import torch


def run(args):
    """Run dodonaphy.
    Using an embedding for MCMC, or embedding variational inference,
    or using a distance matrix for maximum likelihood."""
    if args.path_root == "":
        root_dir = os.path.abspath(".")
    else:
        root_dir = os.path.abspath(args.path_root)

    if args.infer == "simulate":
        simulate_tree(root_dir, args.birth, args.death, args.taxa, args.seq)
        return

    path_write = get_path(root_dir, args)
    dna = read_dna(root_dir, args.path_dna)
    partials, weights, tip_namespace = compress_alignment(dna, get_namespace=True)
    empirical_freqs = compute_nucleotide_frequencies(dna)
    tip_labels = dna.taxon_namespace.labels()
    start_tree = get_start_tree(args.start, dna, root_dir, dna.taxon_namespace)
    if args.location_file is None:
        dists = get_dists(start_tree, matsumoto=args.matsumoto, internals=args.connect == "fix")
    else:
        dists = None
    save_period = max(int(args.epochs / args.draws), 1)
    peel = None
    get_peel = False
    if args.connect == "fix":
        get_peel = True
    if args.use_bito:
        get_peel = True
        msa_file = os.path.join(root_dir, args.path_dna)
        msa_file = os.path.abspath(msa_file)
    if get_peel:
        peel, _, _ = tree.dendropy_to_pb(start_tree)
    
    if args.checkpoint is not None:
        with open(args.checkpoint) as file_pointer:
            checkpoint = json.load(file_pointer, cls=TensorDecoder)

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
            peel=peel,
            embedder=args.embed,
            curvature=args.curv,
            normalise_leaf=args.normalise_leaves,
            loss_fn=args.loss_fn,
            swap_period=args.swap_period,
            n_swaps=args.n_swaps,
            matsumoto=args.matsumoto,
            tip_labels=tip_labels,
            warm_up=args.warm_up,
            mcmc_alg=args.mcmc_alg,
            write_dists=args.write_dists,
            prior=args.prior,
        )

    elif args.infer == "vi":
        mymod = DodonaphyVI(
            partials[:],
            weights,
            args.dim,
            embedder=args.embed,
            connector=args.connect,
            soft_temp=args.temp,
            curvature=args.curv,
            tip_labels=tip_labels,
            n_boosts=args.boosts,
            model_name=args.model,
            freqs=empirical_freqs,
            path_write=path_write,
            prior_fn=args.prior
        )

        # initialise embedding parameters
        mymod.log("%-12s: %s\n" % ("Start Tree", args.start))
        mymod.embed_tree_distribtution(
            dists,
            location_file=args.location_file,
            hydra_max_iter=args.hydra_max_iter,
        )

        if args.use_bito:
            fasta_file = get_fasta_file(msa_file)
            if peel is None:
                raise ValueError("Start tree cannot be None for bito.")
            mymod.init_bito(fasta_file, peel)
        
        if args.checkpoint is not None:
            mymod.set_parameters(checkpoint)

        mymod.learn(
            epochs=args.epochs,
            importance_samples=args.importance,
            n_draws=args.draws,
            lr=args.learn,
        )

    elif args.infer == "hmap":
        mymod = HMAP(
            partials[:],
            weights,
            dim=args.dim,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            path_write=path_write,
            prior=args.prior,
            tip_labels=tip_labels,
            matsumoto=args.matsumoto,
            connector=args.connect,
            peel=peel,
            normalise_leaves=args.normalise_leaves,
            model_name=args.model,
            freqs=empirical_freqs,
            embedder=args.embed,
            curvature=args.curv,
        )
        if args.use_bito:
            fasta_file = get_fasta_file(msa_file)
            if peel is None:
                raise ValueError("Start tree cannot be None for bito.")
            mymod.init_bito(fasta_file, peel)
        if args.location_file is not None:
            args.location_file = os.path.join(root_dir, args.location_file)
        mymod.init_embedding_params(args.location_file, dists, hydra_max_iter=args.hydra_max_iter)

        if args.checkpoint is not None:
            mymod.set_parameters(checkpoint)

        mymod.learn(
            epochs=args.epochs,
            learn_rate=args.learn,
            save_locations=args.write_dists,
            start=args.start,
        )

    elif args.infer == "hlaplace":
        partials, weights = compress_alignment(dna)
        mymod = Laplace(
            partials[:],
            weights,
            dim=args.dim,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            prior=args.prior,
            tip_labels=tip_labels,
            matsumoto=args.matsumoto,
            model_name=args.model,
        )
        mymod.learn(epochs=args.epochs, learn_rate=args.learn, path_write=path_write)
        mymod.laplace(path_write, n_samples=args.draws)

    elif args.infer == "brute":
        partials, weights = compress_alignment(dna)
        path_write = "./test_brute"
        mymod = Brute()
        mymod.run(
            args.dim,
            partials,
            weights,
            path_write,
            dists=dists,
            tip_labels=tip_labels,
            epochs=1000,
            n_boosts=1,
            n_draws=100,
            embedder="up",
            lr=1e-3,
            curvature=-1.0,
            connector="nj",
            soft_temp=None,
        )

    mins, secs = divmod(time.time() - start, 60)
    hrs, mins = divmod(mins, 60)
    print(
        f"Time taken for {args.taxa} taxa with {args.epochs} epochs: {int(hrs)}:{int(mins)}:{int(secs)}\n"
    )


def get_fasta_file(msa_file):
    if not msa_file.endswith(".nex"):
        raise IOError("msa file must end with .nex")
    file_stub = msa_file[:-4]
    fasta_file = file_stub + ".fasta"
    if not os.path.isfile(fasta_file):
        raise FileNotFoundError(f"fasta file {fasta_file} not found. See --help.")
    return fasta_file


def get_start_tree(method, dna, root_dir, taxon_namespace):
    tip_labels = taxon_namespace.labels()
    name_id = {name: id for id, name in enumerate(tip_labels)}
    if method == "NJ":
        print("Computing adjusted distances from tree file:", end="", flush=True)
        dists_hamming = calculate_pairwise_distance(dna, adjust="JC69")
        print(" done.", flush=True)
        peel, blens = Cpeeler.nj_np(dists_hamming)
        nwk = tree.tree_to_newick(name_id, peel, blens)
        start_tree = dendropy.Tree.get(data=nwk, schema="newick")

    elif method == "RAxML":
        print("Finding RAxML tree.")
        rax = raxml.RaxmlRunner()
        start_tree = rax.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
        tree_path = os.path.join(root_dir, "start_tree.nex")
        start_tree.write(path=tree_path, schema="nexus")

    elif method == "Random":
        n_tips = len(dna)
        nC2 = n_tips * (n_tips - 1) / 2
        dists_linear = np.random.exponential(scale=0.1, size=int(nC2))
        dists = np.zeros((n_tips, n_tips))
        dists[np.tril_indices(n_tips, k=-1)] = dists_linear
        dists[np.triu_indices(n_tips, k=+1)] = dists_linear

        peel, blens = Cpeeler.nj_np(dists)
        nwk = tree.tree_to_newick(name_id, peel, blens)
        start_tree = dendropy.Tree.get(data=nwk, schema="newick")
    else:
        start_tree = read_tree(root_dir, taxon_namespace, file_name=method)
    return start_tree


def get_dists(start_tree, matsumoto=False, internals=False):
    if internals:
        dists = utils.all_distances(start_tree)
    else:
        dists = utils.tip_distances(start_tree)
    if matsumoto:
        dists = np.log(np.cosh(dists))
    return dists


def get_path(root_dir, args):
    """Generate and return experiment path"""
    if args.no_save is True:
        return None

    if args.suffix != "":
        args.suffix = "_" + args.suffix
    exp_method = f"{args.embed}_{args.connect}"

    if args.infer == "vi":
        ln_lr = -int(np.log10(args.learn))
        method_dir = os.path.join(root_dir, "vi", exp_method)
        path_write = os.path.join(
            method_dir,
            f"d{args.dim}_lr{ln_lr}_i{args.importance}_b{args.boosts}{args.suffix}",
        )

    elif args.infer == "mcmc":
        method_dir = os.path.join(root_dir, "mcmc", exp_method)
        if args.curv < 0:
            ln_crv = int(np.log10(-args.curv))
        else:
            ln_crv = str(np.log10(-args.curv))
        path_write = os.path.join(method_dir, f"d{args.dim}_k{ln_crv}{args.suffix}")

    elif args.infer in ("hmap", "hlaplace"):
        ln_rate = -int(np.log10(args.learn))
        ln_tau = -int(np.log10(args.temp))
        method_dir = os.path.join(root_dir, args.infer, args.connect, args.prior)
        path_write = os.path.join(method_dir, f"lr{ln_rate}_tau{ln_tau}{args.suffix}")

    if path_write is not None:
        print(f"Saving to {path_write}")
        if not os.path.exists(method_dir):
            try:
                os.makedirs(method_dir, exist_ok=False)
            except OSError:
                print(
                    f"Failed making directoy {method_dir}. Possibly an array job on HPC."
                )
        os.makedirs(path_write, exist_ok=args.overwrite)
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

    tree_info_path = os.path.join(root_dir, "start_tree.log")
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


def read_tree(root_dir, taxon_namespace, file_name="start_tree.nex"):
    """Read a saved nexus or newick tree using dendropy."""
    tree_path = os.path.join(root_dir, file_name)
    try:
        tree = dendropy.Tree.get(
            path=tree_path,
            schema="nexus",
            preserve_underscores=True,
            taxon_namespace=taxon_namespace,
        )
    except:
        tree = dendropy.Tree.get(
            path=tree_path,
            schema="newick",
            preserve_underscores=True,
            taxon_namespace=taxon_namespace,
        )
    return tree


def read_dna(root_dir, file_name="dna.nex"):
    """Get dna from a saved simulated tree."""
    dna_path = os.path.join(root_dir, file_name)
    dna = dendropy.DnaCharacterMatrix.get(
        path=dna_path, schema="nexus", preserve_underscores=True
    )
    return dna


def main():
    """Main entry point"""
    parser = cli.init_parser()
    args = parser.parse_args()
    cli.validate(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    print(f"SEED: {torch.initial_seed()}")
    torch.set_default_dtype(torch.float64)
    run(args)


if __name__ == "__main__":
    main()
