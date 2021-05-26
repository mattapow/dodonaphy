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


def main():
    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Get all pair-wise node distance
    t = Tree(simtree._as_newick_string() + ";")
    nodes = t.get_tree_root().get_descendants(strategy="levelorder")
    dists = [t.get_distance(x, y) for x in nodes for y in nodes]
    dists = np.array(dists).reshape(len(nodes), len(nodes))

    # embed points from distances with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
        emm["r"]), torch.from_numpy(emm["directional"]))
    loc_t0 = p2t0(loc_poin).detach().numpy()

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = Mcmc(partials, weights, dim, loc_t0)

    n_steps = 100
    mymod.learn(n_steps, step_scale=0.001, save_period=10)


if __name__ == '__main__':
    main()
