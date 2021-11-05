import os
import random

import dendropy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dendropy.interop import raxml
from dodonaphy import tree
from dodonaphy.base_model import BaseModel
from dodonaphy.phylo import compress_alignment


def main():
    # Experimental folder
    S = 17  # number of sequences to simulate
    root_dir = os.path.abspath(os.path.join("..", "data", "T%d_hypNJ" % (S)))
    dna_path = os.path.join(root_dir, "dna.nex")

    # load in the dna
    dna = dendropy.DnaCharacterMatrix.get(path=dna_path, schema="nexus")

    # do the bootstrap
    LL = bootstrap(dna, n_samples=100, method='nj')
    LL = LL[LL > -np.inf]

    # plot
    sns.kdeplot(data=LL)
    # plt.hist(LL)
    plt.title('Bootstrap Distribution of Likelihood')
    plt.show()


def bootstrap(dna, n_samples=100, method='nj'):
    """Bootstrap phylogenetic analysis.
    """
    rx = raxml.RaxmlRunner()
    n_sites = len(dna[0])
    idx = np.arange(n_sites)
    LL = np.zeros(n_samples)
    for i in range(n_samples):
        print(f"{i+1}/{n_samples}")
        dna_sample = dna.export_character_indices(random.choices(idx, k=n_sites))
        if method == 'RAxML':
            bs_tree = rx.estimate_tree(char_matrix=dna_sample, raxml_args=["--no-bfgs"])
        elif method == 'nj':
            pdm = get_pdm(dna_sample)
            bs_tree = pdm.nj_tree()
        peel, blen = tree.dendrophy_to_pb(bs_tree)
        partials, weights = compress_alignment(dna_sample)
        model = BaseModel(partials, weights, dim=-1)
        LL[i] = model.compute_LL(peel, blen).numpy()
    return LL


def get_pdm(dna):
    """Get the patristic distance between each pair of taxa.
    """
    n_taxa = len(dna)
    pdm = np.zeros((n_taxa, n_taxa))
    for i in range(n_taxa):
        for j in range(i):
            seq0 = dna.sequences()[i].symbols_as_list()
            seq1 = dna.sequences()[j].symbols_as_list()

            dist_ij = len([i for i, j in zip(seq0, seq1) if i != j])
            pdm[i, j] = pdm[j, i] = dist_ij

    # normalise
    pdm /= dna.sequence_size
    # genetic distance to patristic distance under JC69
    pdm = .75 * (1. - np.exp(-4./3. * pdm))
    fp = "temp.csv"
    np.savetxt(fp, pdm, delimiter=",")
    out = dendropy.PhylogeneticDistanceMatrix.from_csv(
        fp, is_first_row_column_names=False, is_first_column_row_names=False)
    os.remove(fp)
    return out


if __name__ == "__main__":
    main()
