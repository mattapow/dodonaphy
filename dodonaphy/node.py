class Node:
    def __init__(self, taxon=None):
        self.taxon = taxon
        self._nj_distances = {}
        self._nj_xsub = 0.0
