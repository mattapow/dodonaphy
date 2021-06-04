import torch
from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc
from dendropy import Tree as Tree
from dendropy.model.birthdeath import birth_death_likelihood as birth_death_likelihood


class BaseModel(object):
    """Base Model for Inference
    """

    def __init__(self, partials, weights, dim, **prior):
        self.partials = partials
        self.weights = weights
        self.S = len(partials)
        self.L = partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        self.prior = prior

        # make space for internal partials
        for i in range(self.S - 1):
            self.partials.append(torch.zeros((1, 4, self.L), dtype=torch.float64, requires_grad=False))

    def compute_branch_lengths(self, S, D, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
        """Computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball

        Args:
            S (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        blens = torch.empty(self.bcount, dtype=torch.float64)
        for b in range(S-1):
            directional2 = int_dir[peel[b][2]-S-1, ]
            r2 = int_r[peel[b][2]-S-1]

            for i in range(2):
                if peel[b][i] < S:
                    # leaf to internal
                    r1 = leaf_r[peel[b][i]]
                    directional1 = leaf_dir[peel[b][i], :]
                else:
                    # internal to internal
                    r1 = int_r[peel[b][i]-S-1]
                    directional1 = int_dir[peel[b][i]-S-1, ]

                hd = utilFunc.hyperbolic_distance(
                    r1, r2, directional1, directional2, curvature)

                # apply the inverse transform from Matsumoto et al 2020
                hd = torch.log(torch.cosh(hd))

                # add a tiny amount to avoid zero-length branches
                eps = torch.finfo(torch.double).eps
                blens[peel[b][i]] = torch.clamp(hd, min=eps)

        return blens

    def compute_LL(self, leaf_r, leaf_dir, int_r, int_dir):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """

        with torch.no_grad():
            peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)

        # branch lengths
        blens = self.compute_branch_lengths(
            self.S, self.D, peel, leaf_r, leaf_dir, int_r, int_dir)

        mats = JC69_p_t(blens)
        return calculate_treelikelihood(self.partials, self.weights, peel, mats,
                                        torch.full([4], 0.25, dtype=torch.float64))

    def compute_prior(self, leaf_r, leaf_dir, int_r, int_dir, **prior):
        """Calculates the log-likelihood of a tree under a birth death model.

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
            **prior: [description]

        Returns:
            lnl : float

            The log-likehood of the tree under the birth-death model.
        """
        birth_rate = prior.get('birth_rate', 2.)
        death_rate = prior.get('death_rate', .5)

        tipnames = ['T' + str(x+1) for x in range(self.S)]
        peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
        blen = self.compute_branch_lengths(self.S, self.D, peel, leaf_r, leaf_dir, int_r, int_dir)
        newick = utilFunc.tree_to_newick(tipnames, peel, blen)
        tree = Tree.get(data=newick, schema='newick')
        return birth_death_likelihood(
            tree=tree,
            ultrametricity_precision=False,
            birth_rate=birth_rate,
            death_rate=death_rate)
