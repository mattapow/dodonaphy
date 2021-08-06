import torch
from .phylo import calculate_treelikelihood, JC69_p_t
from . import peeler
from . import tree as treeFunc
import Cutils
from dendropy import Tree as Tree
from dendropy.model.birthdeath import birth_death_likelihood as birth_death_likelihood
import numpy as np
import math


class BaseModel(object):
    """Base Model for Inference
    """

    def __init__(self, partials, weights, dim, **prior):
        self.partials = partials.copy()
        self.weights = weights
        self.S = len(self.partials)
        self.L = self.partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        self.prior = prior

        # make space for internal partials
        for i in range(self.S - 1):
            self.partials.append(torch.zeros((1, 4, self.L), dtype=torch.float64, requires_grad=False))

    def initialise_ints(self, emm_tips, n_grids=10, n_trials=10, max_scale=2):
        # try out some inner node positions and pick the best
        # working in the Poincare ball P^d
        print("Randomly initialising internal node positions from {} samples: ".format(n_grids*n_trials),
              end='', flush=True)

        leaf_r = torch.from_numpy(emm_tips['r'])
        leaf_dir = torch.from_numpy(emm_tips['directional'])
        S = len(emm_tips['r'])
        scale = .5 * emm_tips['r'].min()
        lnP = -math.inf
        dir = np.random.normal(0, 1, (S-2, self.D))
        abs = np.sum(dir**2, axis=1)**0.5
        # _int_r = np.random.exponential(scale=scale, size=(S-2))
        # _int_r = scale * np.random.beta(a=2, b=5, size=(S-2))
        _int_r = np.random.uniform(low=0, high=scale, size=(S-2))
        _int_dir = dir/abs.reshape(S-2, 1)
        max_scale = max_scale * emm_tips['r'].min()

        for i in range(n_grids):
            _scale = i/n_grids * max_scale
            for _ in range(n_trials):
                peel = peeler.make_peel_mst(leaf_r, leaf_dir, torch.from_numpy(_int_r), torch.from_numpy(_int_dir))
                blen = self.compute_branch_lengths(
                    self.S, peel, leaf_r, leaf_dir, torch.from_numpy(_int_r), torch.from_numpy(_int_dir))
                _lnP = self.compute_LL(peel, blen)

                if _lnP > lnP:
                    int_r = _int_r
                    int_dir = _int_dir
                    lnP = _lnP
                    scale = _scale

                dir = np.random.normal(0, 1, (S-2, self.D))
                abs = np.sum(dir**2, axis=1)**0.5
                # _int_r = np.random.exponential(scale=_scale, size=(S-2))
                _int_r[_int_r > emm_tips['r'].min()] = emm_tips['r'].max()
                # _int_r = _scale * np.random.beta(a=2, b=5, size=(S-2))
                _int_r = np.random.uniform(low=0, high=_scale, size=(S-2))
                _int_dir = dir/abs.reshape(S-2, 1)

        print("done.\nBest internal node positions selected.")
        if scale > .9 * max_scale:
            print("Using scale=%f, from max of %f. Consider a higher maximum."
                  % (scale/emm_tips['r'].min(), max_scale/emm_tips['r'].min()))

        return int_r, int_dir

    def compute_branch_lengths(self, S, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1), useNP=True):
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
        # Using numpy in cython may be faster
        if useNP:
            DTYPE = np.double
            leaf_r = leaf_r.detach().numpy().astype(DTYPE)
            leaf_dir = leaf_dir.detach().numpy().astype(DTYPE)
            int_r = int_r.detach().numpy().astype(DTYPE)
            int_dir = int_dir.detach().numpy().astype(DTYPE)
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

                if useNP:
                    hd = torch.tensor(Cutils.hyperbolic_distance_np(
                        r1, r2, directional1, directional2, curvature))
                else:
                    hd = Cutils.hyperbolic_distance(
                        r1, r2, directional1, directional2, curvature)

                # apply the inverse transform from Matsumoto et al 2020
                hd = torch.log(torch.cosh(hd))

                # add a tiny amount to avoid zero-length branches
                eps = torch.finfo(torch.double).eps
                blens[peel[b][i]] = torch.clamp(hd, min=eps)

        return blens

    def compute_LL(self, peel, blen):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """
        mats = JC69_p_t(blen)
        return calculate_treelikelihood(self.partials, self.weights, peel, mats,
                                        torch.full([4], 0.25, dtype=torch.float64))

    def compute_log_a_like(self, leaf_r, leaf_dir, curvature=1.):
        """Compute the log-a-like function of the embedding.

        The log-probability of all the pairwise taxa.
        """
        # get pair-wise disatance
        pdm = torch.from_numpy(Cutils.get_pdm(leaf_r, leaf_dir, curvature=curvature, asNumpy=True))

        eps = torch.finfo(torch.double).eps
        P = torch.zeros((4, 4, self.L))

        # For each node
        for i in range(self.S):
            # compute the probability matrix to each other node
            mats = JC69_p_t(pdm[i])
            for j in range(i - 1):
                P = P + torch.log(torch.clamp(torch.matmul(mats[j], self.partials[i]), min=eps))
                # P = P + torch.log(torch.clamp(torch.matmul(mats[j] / pdm[i, j], self.partials[i]), min=eps))

        # normalise for 1/distance weighting
        # idx = torch.triu_indices(self.S, self.S, offset=1)
        # P = P + torch.sum(torch.log(pdm[idx[0], idx[1]]))

        # normalise for number of sites
        return torch.sum(P * self.weights) / sum(self.weights)

    def select_peel_mst(self, leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
        leaf_node_count = leaf_r.shape[0]
        lnP = torch.zeros(leaf_node_count)
        # Randomly select leaves if getting slow
        for leaf in range(leaf_node_count):
            peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1), start_node=leaf)
            blens = self.compute_branch_lengths(leaf_node_count, peel, leaf_r, leaf_dir, int_r, int_dir)
            print("LL=%f, Length=%f" % (float(self.compute_LL(peel, blens)), float(sum(blens))))
            lnP[leaf] = self.compute_LL(peel, blens)
        print("")
        sftmx = torch.nn.Softmax(dim=0)
        p = np.array(sftmx(lnP))
        leaf = np.random.choice(leaf_node_count, p=p)
        print(p)
        return peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1), start_node=leaf)

    @staticmethod
    def compute_prior_birthdeath(peel, blen, **prior):
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
        S = len(blen) / 2 + 1

        birth_rate = prior.get('birth_rate', 2.)
        death_rate = prior.get('death_rate', .5)

        tipnames = ['T' + str(x+1) for x in range(S)]
        newick = treeFunc.tree_to_newick(tipnames, peel, blen)
        tree = Tree.get(data=newick, schema='newick')
        LL = birth_death_likelihood(
            tree=tree,
            ultrametricity_precision=False,
            birth_rate=birth_rate,
            death_rate=death_rate)
        return torch.tensor(LL)

    @staticmethod
    def compute_prior_gamma_dir(blen, aT=torch.ones(1), bT=torch.full((1,), .1), a=torch.ones(1),
                                c=torch.ones(1)):
        """Compute prior under a gamma-Dirichlet(αT , βT , α, c) prior.

        Rannala et al., 2012; Zhang et al., 2012
        Following MrBayes:
        "The prior assigns a gamma(αT , βT ) distribution for the tree length
        (sum of branch lengths), and a Dirichlet(α, c) prior for the proportion
        of branch lengths to the tree length. In the Dirichlet, the parameter for
        external branches is α and for internal branches is αc, so that the prior
        ratio between internal and external branch is c."
        NB: scaling constants from Rannala et al., 2020 are omitted, as for MrBayes.

        Args:
            blen ([type]): [description]
            aT ([type], optional): [description]. Defaults to torch.ones(1).
            bT ([type], optional): [description]. Defaults to torch.full((1,), .1).
            a ([type], optional): [description]. Defaults to torch.ones(1).
            c ([type], optional): [description]. Defaults to torch.ones(1).

        Returns:
            tensor: The log probability of the branch lengths under the prior.
        """
        bcount = int(len(blen))
        S = int(bcount / 2 + 1)

        lnprior = torch.zeros(1)
        treeL = sum(blen)
        tipb = torch.log(sum(blen[:S]))
        intb = torch.log(sum(blen[S:]))

        lnprior = lnprior + (a-1)*tipb + (a*c-1)*intb
        lnprior = lnprior + (aT - a*S - a*c*(S-3)) * torch.log(treeL) - bT*treeL

        return lnprior
