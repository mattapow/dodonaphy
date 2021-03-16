import sys

from dendropy import DnaCharacterMatrix
from dodonaphy.model import DodonaphyModel
from dodonaphy.phylo import compress_alignment


def testFunc(alignment_path, dimension):
    dna = DnaCharacterMatrix.get(path=alignment_path, schema='fasta')
    partials, weights = compress_alignment(dna)
    mymod = DodonaphyModel(partials, weights, dimension)
    mymod.learn()


testFunc(sys.argv[1], int(sys.argv[2]))
