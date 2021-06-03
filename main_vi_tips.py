import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.vi import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from src.hyperboloid import p2t0
# from matplotlib import pyplot as plt
# import matplotlib.cm
import numpy as np
import torch
import random
import os


def main():
    """
    Initialise the emebedding with tips distances given to hydra.
    Internal nodes are in distributions at origin.
    """

    dim = 2  # number of dimensions for embedding
    S = 6  # number of sequences to simulate
    L = 1000  # length of sequences to simulate

    # simulate a tree
    rng = random.Random(1)
    simtree = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.5, num_extant_tips=S, rng=rng)
    dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69(), rng=rng)

    # save dna to nexus
    path_write = "./out"
    os.mkdir(path_write)  # os.makedirs(path_write, exist_ok=True)
    dest = path_write + "/dna.nex"
    dna.write_to_path(dest, "nexus")

    # Get tip pair-wise distance
    dists = np.zeros((S, S))
    pdc = simtree.phylogenetic_distance_matrix()
    for i, t1 in enumerate(simtree.taxon_namespace[:-1]):
        for j, t2 in enumerate(simtree.taxon_namespace[i+1:]):
            dists[i][i+j+1] = pdc(t1, t2)
    dists = dists + dists.transpose()

    # embed with hydra
    emm = utilFunc.hydra(D=dists, dim=dim, equi_adj=0., stress=True)
    print('stress = ' + str(emm["stress"]))

    leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(
        emm["r"]), torch.from_numpy(emm["directional"]))
    leaf_loc_t0 = p2t0(leaf_loc_poin).detach().numpy()

    # set initial leaf positions from hydra with small coefficient of variation
    # set internal nodes to narrow distributions at origin
    cv = 1. / 50
    eps = np.finfo(np.double).eps
    leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
    param_init = {
        "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
        "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
        "int_mu": torch.zeros(S - 2, dim, requires_grad=True, dtype=torch.float64),
        "int_sigma": torch.full((S - 2, dim), np.log(.01), requires_grad=True, dtype=torch.float64)
    }

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dim)

    # learn
    mymod.learn(param_init=param_init, epochs=10)

    # draw samples
    nsamples = 2
    peels, blens, X, lp = mymod.draw_sample(nsamples, lp=True)

    # # Plot embedding if dim==2
    # if dim == 2:
    #     _, ax = plt.subplots(1, 1)
    #     ax.set(xlim=[-1, 1])
    #     ax.set(ylim=[-1, 1])
    #     cmap = matplotlib.cm.get_cmap('Spectral')
    #     for i in range(nsamples):
    #         utilFunc.plot_tree(ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
    #     ax.set_title("Final Embedding Sample")
    #     plt.show()

    fn = path_write + '/vi.trees'
    with open(fn, 'w') as file:
        file.write("#NEXUS\n\n")
        file.write("Begin taxa;\n\tDimensions ntax=" + str(S) + ";\n")
        file.write("\tTaxlabels\n")
        for i in range(S):
            file.write("\t\t" + "T" + str(i+1) + "\n")
        file.write("\t\t;\nEnd;\n\n")
        file.write("Begin trees;\n")

    with open(fn, 'a') as file:
        for i in range(nsamples):
            utilFunc.save_tree(path_write, "vi", peels[i], blens[i], i, 0)


if __name__ == "__main__":
    main()
