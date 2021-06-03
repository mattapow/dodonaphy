from src.vi import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from src.hyperboloid import p2t0

import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from matplotlib import pyplot as plt
import matplotlib.cm
import numpy as np
import torch
from ete3 import Tree


def main():
    """
    Initialise int and lead embeddings with hydra
    Optimise VM
    Plot samples from VM
    """

    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # Get all pair-wise node distance
    t = Tree(simtree._as_newick_string() + ";")
    nodes = t.get_tree_root().get_descendants(strategy="levelorder")
    dists = [t.get_distance(x, y) for x in nodes for y in nodes]
    dists = np.array(dists).reshape(len(nodes), len(nodes))

    # embed points with Hydra
    emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0)
    loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))
    loc_t0 = p2t0(loc_poin)
    leaf_loc_t0 = loc_t0[:S, :].detach().numpy()
    int_loc_t0 = loc_t0[S:, :].detach().numpy()

    # set initial node positions from hydra with small coefficient of variation
    cv = 0.1
    eps = np.finfo(np.double).eps
    leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
    int_sigma = np.log(np.abs(np.array(int_loc_t0)) * cv + eps)
    param_init = {
        "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
        "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
        "int_mu": torch.tensor(int_loc_t0, requires_grad=True, dtype=torch.float64),
        "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64)
    }

    # learn
    mymod.learn(param_init=param_init, epochs=100)

    if dim == 2:
        # sample
        nsamples = 3
        peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

        # plot tree
        _, ax = plt.subplots(1, 1)
        ax.set(xlim=[-1, 1])
        ax.set(ylim=[-1, 1])
        cmap = matplotlib.cm.get_cmap('Spectral')
        for i in range(nsamples):
            utilFunc.plot_tree(ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
        ax.set_title("Final Embedding Sample")
        plt.show()


if __name__ == "__main__":
    main()
