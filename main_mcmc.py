from src.mcmc import Mcmc
from src.phylo import compress_alignment
from src.utils import utilFunc
from src.hyperboloid import p2t0

import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from ete3 import Tree
import numpy as np
import torch
import random
from dendropy.interop import raxml
import os
import math


def main():
    dim = 2  # number of dimensions for embedding
    S = 10  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # Simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(birth_rate=2., death_rate=0.5, num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())
    # compress alignment
    partials, weights = compress_alignment(dna)

    # save dna to nexus
    path_write = "./out"
    os.mkdir(path_write)  # os.makedirs(path_write, exist_ok=True)
    dest = path_write + "/dna.nex"
    dna.write_to_path(dest, "nexus")

    all_dists = False
    if all_dists:
        # Get all pair-wise node distance
        t = Tree(simtree._as_newick_string() + ";")
        nodes = t.get_tree_root().get_descendants(strategy="levelorder")
        dists = [t.get_distance(x, y) for x in nodes for y in nodes]
        dists = np.array(dists).reshape(len(nodes), len(nodes))

        # embed points from distances with Hydra
        emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0, stress=True)
        print("Embedding Stress = {:.4}".format(emm["stress"].item()))
    else:
        # Get tip pair-wise distance from RAxML tree
        rx = raxml.RaxmlRunner()
        rxml_tree = rx.estimate_tree(char_matrix=dna)
        assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
        dists = np.array([[assemblage_data[i][j] for j in sorted(
            assemblage_data[i])] for i in sorted(assemblage_data)])

        # embed points from distances with Hydra
        emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0, stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(emm["stress"].item()))

        # internal nodes near origin
        int_r, int_dir = initialise(emm, dim, partials, weights)
        emm["r"] = np.concatenate((emm["r"], int_r))
        emm["directional"] = np.concatenate((emm["directional"], int_dir))

    # store in tangent plane R^dim
    loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))
    loc_t0 = p2t0(loc_poin)

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = Mcmc(partials, weights, dim, loc_t0)

    # Learn
    epochs = 1000
    mymod.learn(epochs, path_write=path_write, step_scale=0.01, save_period=1)


def initialise(emm_tips, dim, partials, weights):
    # try out some inner node positions and pick the best
    print("Randomly initialising internal node positions.")

    S = len(emm_tips['r'])
    scale = torch.as_tensor(.5 * emm_tips['r'].min())
    mod = Mcmc(partials, weights, dim, None)
    lnP = -math.inf

    dir = np.random.normal(0, 1, (S-2, dim))
    abs = np.sum(dir**2, axis=1)**0.5
    _int_r = np.random.exponential(scale=scale, size=(S-2))
    _int_dir = dir/abs.reshape(S-2, 1)

    n_scale = 10
    max_scale = 5 * emm_tips['r'].min()
    n_trials = 100
    for i in range(n_scale):
        _scale = torch.as_tensor((i+1)/(n_scale+1) * max_scale)
        for _ in range(n_trials):
            _lnP = mod.compute_LL(
                torch.from_numpy(emm_tips['r']), torch.from_numpy(emm_tips['directional']),
                torch.from_numpy(_int_r), torch.from_numpy(_int_dir))

            if _lnP > lnP:
                int_r = _int_r
                int_dir = _int_dir
                lnP = _lnP
                scale = _scale

            dir = np.random.normal(0, 1, (S-2, dim))
            abs = np.sum(dir**2, axis=1)**0.5
            _int_r = np.random.exponential(scale=_scale, size=(S-2))
            _int_dir = dir/abs.reshape(S-2, 1)

    print("Best internal node positions found.")

    return int_r, int_dir


if __name__ == '__main__':
    with torch.no_grad():
        main()
