import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.model import DodonaphyModel
from src.phylo import compress_alignment
from src.utils import utilFunc
from matplotlib import pyplot as plt
import matplotlib.cm

"""Testing Dodonaphy model with randomly initialized parameters for variational inference
"""
dim = 3    # number of dimensions for embedding
nseqs = 6  # number of sequences to simulate
seqlen = 1000  # length of sequences to simulate

# simulate a tree
simtree = treesim.birth_death_tree(
    birth_rate=2.0, death_rate=0.5, num_extant_tips=nseqs)
dna = simulate_discrete_chars(
    seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

# testing raxml
# rx = raxml.RaxmlRunner(raxml_path="raxmlHPC-AVX2")
# # tree = rx.estimate_tree(char_matrix=dna, raxml_args=['-e', 'likelihoodEpsilon', '-h' '--JC69'])
# tree = rx.estimate_tree(char_matrix=dna, raxml_args=["-h", "--JC69"])

# rx = raxml.RaxmlRunner()
# rxml_tree = rx.estimate_tree(char_matrix=dna)
# assemblage_data = rxml_tree.phylogenetic_distance_matrix().as_data_table()._data
# dist = np.array([[assemblage_data[i][j] for j in sorted(
#     assemblage_data[i])] for i in sorted(assemblage_data)])
# emm = utilFunc.hydra(D=dist, dim=dim)

# model initiation and training
partials, weights = compress_alignment(dna)
mymod = DodonaphyModel(partials, weights, dim)
# variational parameters: [default] randomly generated within model constructor
mymod.learn(epochs=10)

# draw samples from variational posterior
nsamples = 3
peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)

# compare dodonapy with RAxML
# tip_labels = simtree.taxon_namespace.labels()
# rxml_peel, rxml_blens = utilFunc.dendrophy_to_pb(rxml_tree)
# rxml_tree_nw = utilFunc.tree_to_newick(tip_labels, rxml_peel, rxml_blens)
# rxml_peel_dp = dendropy.Tree.get(data=rxml_tree_nw, schema="newick")
dodonaphy_tree_nw = utilFunc.tree_to_newick(
    simtree.taxon_namespace.labels(), peels[0], blens[0])
dodonaphy_tree_dp = dendropy.Tree.get(
    data=dodonaphy_tree_nw, schema="newick")
dodonaphy_tree_dp.print_plot()
# dodonaphy_tree_dp = dendropy.TreeList(taxon_namespace=rxml_tree.taxon_namespace)

# # compare raxml and dodonaphy tree based on euclidean and Robinson_foulds distance
# ec_dist = treecalc.euclidean_distance(rxml_peel_dp, dodonaphy_tree_dp)
# rf_dist = treecalc.robinson_foulds_distance(rxml_tree, dodonaphy_tree_dp)

# draw the tree samples
if dim == 2:
    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(1, 1, 1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i in range(nsamples):
        utilFunc.plot_tree(
            ax, peels[i], X[i].detach().numpy(), color=cmap(i / nsamples))
    plt.show()
