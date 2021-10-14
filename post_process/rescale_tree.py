import dendropy
from dendropy.model.birthdeath import birth_death_likelihood
from dendropy.model.discrete import simulate_discrete_chars

S = 17
L = 1000
path_write = "./data/T%d" % (S)
treePath = "%s/simtree.nex" % path_write
dnaPath = "%s/dna.nex" % path_write
treeInfoPath = "%s/simtree.info" % path_write


# load tree
simtree = dendropy.Tree.get(path=treePath, schema="nexus")

# scale
simtree.scale_edges(0.1)

# generate new dna
dna = simulate_discrete_chars(seq_len=L, tree_model=simtree, seq_model=dendropy.model.discrete.Jc69())

# save simtree
simtree.write(path=treePath, schema="nexus")

# save dna to nexus
dna.write_to_path(dest=dnaPath, schema="nexus")


# save simTree info log-likelihood
LL = birth_death_likelihood(tree=simtree, birth_rate=2, death_rate=.5)
with open(treeInfoPath, 'w') as f:
    f.write('Log Likelihood: %f\n' % LL)
    simtree.write_ascii_plot(f)
