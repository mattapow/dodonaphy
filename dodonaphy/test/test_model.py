import dendropy
from dendropy.simulate import treesim
from dendropy.model.discrete import simulate_discrete_chars
from src.vi import DodonaphyVI as vi
from src.phylo import compress_alignment, JC69_p_t, calculate_treelikelihood
from src import tree as treeFunc
import torch
from dendropy.interop import raxml


def test_draws_different():
    """
    Each draw from the sample should be different in likelihood.

    """
    dim = 2  # number of dimensions for embedding
    nseqs = 6  # number of sequences to simulate
    seqlen = 1000  # length of sequences to simulate

    # # simulate a tree
    simtree = treesim.birth_death_tree(
        birth_rate=1.0, death_rate=0.5, num_extant_tips=nseqs)
    dna = simulate_discrete_chars(
        seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Initialise model
    partials, weights = compress_alignment(dna)
    mymod = vi(partials, weights, dim)

    # learn
    mymod.learn(epochs=1, path_write=None)

    # draw
    nsamples = 3
    peels, blens, X, lp__ = mymod.draw_sample(nsamples, lp=True)
    assert not torch.equal(blens[0], blens[1])
    assert not torch.equal(blens[0], blens[2])
    assert not torch.equal(blens[1], blens[2])

    assert not torch.equal(lp__[0], lp__[1])
    assert not torch.equal(lp__[0], lp__[2])
    assert not torch.equal(lp__[1], lp__[2])


def test_calculate_likelihood():
    """
    Sometimes calculate_likelihood was throwing errors about
    torch.matmul(Tensor, list), where it wanted torch.matmul(Tensor, Tensor)
    """

    S = 4  # number of sequences to simulate
    seqlen = 100  # length of sequences to simulate

    # Simulate a tree
    simtree = treesim.birth_death_tree(birth_rate=2., death_rate=0.5, num_extant_tips=S)
    dna = simulate_discrete_chars(seq_len=seqlen, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

    # Compute RAxML tree
    rx = raxml.RaxmlRunner()
    tree = rx.estimate_tree(char_matrix=dna, raxml_args=["--no-bfgs"])
    peel, blens = treeFunc.dendrophy_to_pb(tree)
    mats = JC69_p_t(blens)

    # compute partials and weights
    partials, weights = compress_alignment(dna)
    # make space for internal partials
    for i in range(S - 1):
        partials.append(torch.zeros((1, 4, seqlen), dtype=torch.float64))

    _ = calculate_treelikelihood(partials, weights, peel, mats,
                                 torch.full([4], 0.25, dtype=torch.float64))
