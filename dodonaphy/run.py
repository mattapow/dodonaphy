"""Run Dodonaphy module"""
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
from dodonaphy.map import MAP
from dodonaphy.hmap import HMAP
from dodonaphy.phylo import compress_alignment, calculate_pairwise_distance
from dodonaphy.vi import DodonaphyVI
from dodonaphy.brute import Brute


def run(args):
    """Run dodonaphy.
    Using an embedding for MCMC, or embedding variational inference,
    or using a distance matrix for maximum likelihood."""
    if args.path_root == "":
        root_dir = os.path.abspath("./analysis")
    else:
        root_dir = os.path.abspath(args.path_root)

    if args.infer == "simulate":
        simulate_tree(root_dir, args.birth, args.death, args.taxa, args.seq)
        return

    path_write = get_path(root_dir, args)
    dna = read_dna(root_dir, args.path_dna)
    partials, weights, tip_namespace = compress_alignment(dna, get_namespace=True)
    tip_labels = tip_namespace.labels()

    dists, start_tree = get_start_dists(
        args.start, dna, root_dir, tip_namespace, args.matsumoto
    )
    save_period = max(int(args.epochs / args.draws), 1)
    if args.connect == "fix":
        warnings.warn("Fixed topology is experimental and start tree must have integer taxa names.")
        tree.rename_labels(start_tree)
        peel, _ = tree.dendropy_to_pb(start_tree)
    else:
        peel = None

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
        DodonaphyVI.run(
            args.dim,
            partials[:],
            weights,
            dists,
            path_write,
            epochs=args.epochs,
            importance_samples=args.importance,
            n_draws=args.draws,
            lr=args.learn,
            embedder=args.embed,
            connector=args.connect,
            curvature=args.curv,
            soft_temp=args.temp,
            tip_labels=tip_labels,
            n_boosts=args.boosts,
            start=args.start,
        )

    elif args.infer == "dmap":
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

    elif args.infer == "hmap":
        partials, weights = compress_alignment(dna)
        mymod = HMAP(
            partials[:],
            weights,
            dim=args.dim,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            path_write=path_write,
            prior=args.prior,
            tip_labels=tip_labels,
            matsumoto=args.matsumoto,
            connector=args.connect,
            peel=peel,
            normalise_leaves=args.normalise_leaves
        )
        mymod.learn(
            epochs=args.epochs,
            learn_rate=args.learn,
            save_locations=args.write_dists,
            start=args.start,
        )

    elif args.infer == "hlaplace":
        partials, weights = compress_alignment(dna)
        mymod = HMAP(
            partials[:],
            weights,
            dim=args.dim,
            dists=dists,
            soft_temp=args.temp,
            loss_fn=args.loss_fn,
            prior=args.prior,
            tip_labels=tip_labels,
            matsumoto=args.matsumoto,
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


def get_start_dists(method, dna, root_dir, taxon_namespace, matsumoto=False):
    n_taxa = len(dna)
    start_tree = None
    if method == "None":
        print("Computing adjusted distances from sequences:", end="", flush=True)
        dists = calculate_pairwise_distance(dna, adjust="JC69")
        print(" done.", flush=True)
    elif method == "NJ":
        print("Computing raw distances from tree file:", end="", flush=True)
        dists_hamming = calculate_pairwise_distance(dna, adjust=None)
        print(" done.", flush=True)
        peel, blens = Cpeeler.nj_np(dists_hamming)
        tipnames = [str(i) for i in range(n_taxa)]
        nwk = tree.tree_to_newick(tipnames, peel, blens)
        start_tree = dendropy.Tree.get(data=nwk, schema="newick")
        dists = utils.tip_distances(start_tree, n_taxa)
    elif method == "RAxML":
        print("Finding RAxML tree.")
        rax = raxml.RaxmlRunner()
        start_tree = rax.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
        tree_path = os.path.join(root_dir, "start_tree.nex")
        start_tree.write(path=tree_path, schema="nexus")
        dists = utils.tip_distances(start_tree, n_taxa)
    elif method == "Random":
        n_tips = len(dna)
        nC2 = n_tips * (n_tips - 1) / 2
        dists_linear = np.random.exponential(scale=0.1, size=int(nC2))
        dists = np.zeros((n_tips, n_tips))
        dists[np.tril_indices(n_tips, k=-1)] = dists_linear
        dists[np.triu_indices(n_tips, k=+1)] = dists_linear
    else:
        start_tree = read_tree(root_dir, taxon_namespace, file_name=method)
        dists = utils.tip_distances(start_tree, n_taxa)

    return dists, start_tree


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

    elif args.infer in ("dmap", "hmap", "hlaplace"):
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
    """Read a saved nexus tree using dendropy."""
    tree_path = os.path.join(root_dir, file_name)
    return dendropy.Tree.get(path=tree_path,
        schema="nexus",
        preserve_underscores=True,
        taxon_namespace=taxon_namespace)


def read_dna(root_dir, file_name="dna.nex"):
    """Get dna from a saved simulated tree."""
    dna_path = os.path.join(root_dir, file_name)
    dna = dendropy.DnaCharacterMatrix.get(path=dna_path, schema="nexus")
    return dna


def main():
    """Main entry point"""
    parser = cli.init_parser()
    args = parser.parse_args()
    cli.validate(args)
    run(args)


if __name__ == "__main__":
    main()
