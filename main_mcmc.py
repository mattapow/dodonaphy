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


def main():
    dim = 2  # number of dimensions for embedding
    S = 10  # number of sequences to simulate
    seqlen = 10000  # length of sequences to simulate

    # Simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(birth_rate=2., death_rate=0.5, num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

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
        scale = .1 * emm["r"].min()
        emm["r"] = np.concatenate((emm["r"], np.random.exponential(scale=scale, size=(S-2))))
        u = np.random.normal(0, 1, (S-2, dim))
        norm = np.sum(u**2, axis=1)**0.5
        emm["directional"] = np.concatenate((emm["directional"], u/norm.reshape(S-2, 1)))

    # store in tangent plane R^dim
    loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))
    loc_t0 = p2t0(loc_poin)

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = Mcmc(partials, weights, dim, loc_t0)

    # Learn
    epochs = 1000
    mymod.learn(epochs, path_write=path_write, step_scale=0.03, save_period=1, showPlot=False)

    # path = './out/mcmc.tree'
    # treelist = dendropy.TreeList.get(path=path, schema='newick')
    # for tree in trees:


if __name__ == '__main__':
    with torch.no_grad():
        main()
