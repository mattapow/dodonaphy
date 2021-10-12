from src import poincare
from .phylo import calculate_treelikelihood, JC69_p_t
from . import peeler, hyperboloid, utils
from . import tree as treeFunc
from .utils import LogDirPrior
import Cutils

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from dendropy import Tree as Tree
from dendropy.model.birthdeath import birth_death_likelihood as birth_death_likelihood
import numpy as np
import math


class BaseModel(object):
    """Base Model for Inference
    """

    def __init__(self, partials, weights, dim, curvature=-1., dists_data=None, **prior):
        self.partials = partials.copy()
        self.weights = weights
        self.S = len(self.partials)
        self.L = self.partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        self.prior = prior
        assert curvature <= 0
        self.curvature = torch.tensor(curvature)
        self.epoch = 0
        self.dists_data = dists_data

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

    @staticmethod
    def compute_branch_lengths(S, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1), useNP=True):
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
        bcount = 2*S - 2
        blens = torch.empty(bcount, dtype=torch.float64)
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

    def compute_log_a_like(self, pdm, temp=1.0):
        """Compute the log-a-like function of the embedding.

        The log-probability of all the pairwise taxa.
        """
        eps = torch.finfo(torch.double).eps
        P = torch.zeros((4, 4, self.L))

        Q = peeler.compute_Q(pdm)
        weight = torch.softmax(Q / temp, dim=1)

        # For each node
        for i in range(self.S):
            # compute the probability matrix to each other node
            mats = JC69_p_t(pdm[i])
            for j in range(i - 1):
                P = P + weight[i, j] * torch.log(torch.clamp(torch.matmul(mats[j], self.partials[i]), min=eps))

        L = torch.sum(self.weights)
        return torch.sum(P * self.weights)/L

    def compute_hypHC(self, leaf_X, temperature=0.05, n_triplets=100):
        """Computes log of HypHC loss
        "From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering"

        Args:
            leaf_X  ([type]): Positions of leaves

        Returns:
            [type]: Log of loss
        """
        triplets = similarities = torch.zeros(n_triplets, 3)
        triplets = triplets.long()
        for i in range(n_triplets):
            triplets[i, :] = torch.multinomial(torch.ones((self.S,)), 3)
            similarities[i, :] = torch.tensor(
                self.get_similarities(triplets[i, [0, 0, 1]], triplets[i, [1, 2, 2]], self.dists_data))

        triplets = np.array(triplets)
        e1 = leaf_X[triplets[:, 0]]
        e2 = leaf_X[triplets[:, 1]]
        e3 = leaf_X[triplets[:, 2]]
        d_12 = poincare.hyp_lca(e1, e2, return_coord=False)
        d_13 = poincare.hyp_lca(e1, e3, return_coord=False)
        d_23 = poincare.hyp_lca(e2, e3, return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / temperature, dim=-1)
        w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
        total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)

    def get_similarities(self, u, v, pdm_data,
                         freqs=torch.full([4], 0.25, dtype=torch.float64)):
        """Similarities of taxa u and v.
        Take exp(-pdm)

        Args:
            u ([type]): [description]
            v ([type]): [description]
            pdm_data ([type]): Pairwise distance of data
            freqs ([type], optional): [description]. Defaults to torch.full([4], 0.25, dtype=torch.float64).

        Returns:
            [type]: [description]
        """
        return torch.exp(-pdm_data[u, v])

    def select_peel_mst(self, leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1)):
        leaf_node_count = leaf_r.shape[0]
        lnP = torch.zeros(leaf_node_count)
        # TODO: Randomly select leaves if getting slow
        for leaf in range(leaf_node_count):
            peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1), start_node=leaf)
            blens = self.compute_branch_lengths(leaf_node_count, peel, leaf_r, leaf_dir, int_r, int_dir)
            lnP[leaf] = self.compute_LL(peel, blens)
        sftmx = torch.nn.Softmax(dim=0)
        p = np.array(sftmx(lnP))
        leaf = np.random.choice(leaf_node_count, p=p)
        return peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir, curvature=-torch.ones(1), start_node=leaf)

    def sample(self, leaf_loc, leaf_cov, int_loc=None, int_cov=None, getPeel=True):
        # reshape covariance if single number
        n_leaf_vars = self.S * self.D
        if torch.numel(leaf_cov) == 1:
            leaf_cov = torch.eye(n_leaf_vars, dtype=torch.double) * leaf_cov
        n_int_vars = (self.S-2) * self.D
        if int_cov is not None and torch.numel(int_cov) == 1:
            int_cov = torch.eye(n_int_vars, dtype=torch.double) * int_cov

        if self.embed_method == 'simple':
            # transform leaves to R^n
            leaf_loc_t0 = utils.ball2real(leaf_loc)
            log_abs_det_jacobian = -Cutils.real2ball_LADJ(leaf_loc_t0)

            # flatten data to sample
            leaf_loc_t0 = leaf_loc_t0.reshape(n_leaf_vars)

            # propose new leaf nodes from normal in R^n
            normal_dist = MultivariateNormal(leaf_loc_t0, leaf_cov)
            sample = normal_dist.rsample()
            logQ = normal_dist.log_prob(sample)
            leaf_loc_t0 = sample.reshape((self.S, self.D))

            # # normalise leaves to sphere with radius leaf_r_prop = first leaf radii
            # leaf_r_t0 = torch.norm(leaf_loc_t0[0, :])
            # leaf_loc_t0 = utils.normalise(leaf_loc_t0) * leaf_r_t0

            # Convert to Ball
            leaf_loc_prop = utils.real2ball(leaf_loc_t0)

        elif self.embed_method == 'wrap':
            # transform leaves to R^n
            leaf_loc_t0, log_abs_det_jacobian = hyperboloid.p2t0(leaf_loc, get_jacobian=True)
            leaf_loc_t0 = leaf_loc_t0.clone()

            # propose new leaf nodes from normal in R^n and convert to poincare ball
            normal_dist = MultivariateNormal(torch.zeros((self.S * self.D), dtype=torch.double), leaf_cov)
            sample = normal_dist.rsample()
            logQ = normal_dist.log_prob(sample)
            leaf_loc_prop = hyperboloid.t02p(sample.reshape(self.S, self.D), leaf_loc_t0.reshape(self.S, self.D))

        # get r and directional
        leaf_r_prop = torch.norm(leaf_loc_prop[0, :])
        leaf_dir_prop = leaf_loc_prop / torch.norm(leaf_loc_prop, dim=-1, keepdim=True)

        # internal nodes for mst
        if self.connect_method in ('mst', 'mst_choice'):
            if self.embed_method == 'simple':
                # transform internals to R^n
                int_loc_t0 = utils.ball2real(int_loc)
                log_abs_det_jacobian = log_abs_det_jacobian - Cutils.real2ball_LADJ(int_loc_t0)

                # flatten data to sample
                int_loc_t0 = int_loc_t0.reshape(n_int_vars)

                # propose new int nodes from normal in R^n
                normal_dist = MultivariateNormal(int_loc_t0.squeeze(), int_cov)
                sample = normal_dist.rsample()
                logQ = logQ + normal_dist.log_prob(sample)
                int_loc_t0 = sample.reshape((self.S-2, self.D))

                # convert ints to poincare ball
                int_loc_prop = utils.real2ball(int_loc_t0)

            elif self.embed_method == 'wrap':
                # transform ints to R^n
                int_loc_t0, int_jacobian = hyperboloid.p2t0(int_loc, get_jacobian=True).clone()
                log_abs_det_jacobian = log_abs_det_jacobian - int_jacobian

                # propose new int nodes from normal in R^n
                normal_dist = MultivariateNormal(torch.zeros_like(int_loc_t0.squeeze()), int_cov)
                sample = normal_dist.rsample()
                logQ = logQ + normal_dist.log_prob(sample)
                int_loc_prop = hyperboloid.t02p(sample.reshape(self.S-2, self.D), int_loc_t0.reshape(self.S-2, self.D))

            # get r and directional
            int_r_prop = torch.norm(int_loc_prop, dim=-1)
            int_dir_prop = int_loc_prop / torch.norm(int_loc_prop, dim=-1, keepdim=True)

            # restrict int_r to less than leaf_r
            int_r_prop[int_r_prop > leaf_r_prop] = leaf_r_prop

        # internal nodes and peel for geodesics
        leaf_r = leaf_r_prop.repeat(self.S)
        if self.connect_method in ('geodesics', 'incentre'):
            with torch.no_grad():
                peel, int_locs = peeler.make_peel_tips(
                    leaf_r_prop * leaf_dir_prop, connect_method=self.connect_method)
                int_r_prop, int_dir_prop = utils.cart_to_dir(int_locs)

        # proposal peel for other methods if requested
        pdm = Cutils.get_pdm_torch(leaf_r_prop.repeat(self.S), leaf_dir_prop, curvature=self.curvature)
        if not getPeel:
            peel = blens = lnP = lnPrior = None
        else:
            if self.connect_method == 'nj':
                peel, blens = peeler.nj(pdm)
            elif self.connect_method == 'mst':
                peel = peeler.make_peel_mst(leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)
            elif self.connect_method == 'mst_choice':
                peel = self.select_peel_mst(leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)
            elif self.connect_method == 'delaunay':
                peel = peeler.make_peel_delaunay(leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)

            # get proposal branch lengths
            if self.connect_method != 'nj':
                blens = self.compute_branch_lengths(
                    self.S, peel, leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop, useNP=False)

            # get log likelihood
            lnP = self.compute_LL(peel, blens)
            # lnP = self.compute_log_a_like(pdm)
            # leaf_X = utils.dir_to_cart(leaf_r_prop, leaf_dir_prop)
            # lnP = self.compute_hypHC(leaf_X)

            # get log prior
            lnPrior = self.compute_prior_gamma_dir(blens)
            # lnPrior = self.compute_prior_birthdeath(peel, blens, **self.prior)

        if self.connect_method in ('nj'):
            proposal = {
                'leaf_r': leaf_r_prop,
                'leaf_dir': leaf_dir_prop,
                'peel': peel,
                'blens': blens,
                'jacobian': log_abs_det_jacobian,
                'logQ': logQ,
                'lnP': lnP,
                'lnPrior': lnPrior
            }
        elif self.connect_method in ('geodesics', 'incentre', 'mst', 'mst_choice'):
            proposal = {
                'leaf_r': leaf_r_prop,
                'leaf_dir': leaf_dir_prop,
                'int_r': int_r_prop,
                'int_dir': int_dir_prop,
                'peel': peel,
                'blens': blens,
                'jacobian': log_abs_det_jacobian,
                'logQ': logQ,
                'lnP': lnP,
                'lnPrior': lnPrior
            }
        return proposal

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

        Args:
            blen ([type]): [description]
            aT ([type], optional): [description]. Defaults to torch.ones(1).
            bT ([type], optional): [description]. Defaults to torch.full((1,), .1).
            a ([type], optional): [description]. Defaults to torch.ones(1).
            c ([type], optional): [description]. Defaults to torch.ones(1).

        Returns:
            tensor: The log probability of the branch lengths under the prior.
        """
        n_branch = int(len(blen))
        n_leaf = int(n_branch / 2 + 1)

        # Dirichlet prior
        lnPrior = LogDirPrior(blen, aT, bT, a, c)

        # with prefactor
        lgamma = torch.lgamma
        lnPrior = lnPrior + (aT) * torch.log(bT) - lgamma(aT) + lgamma(a * n_leaf + a * c * (n_leaf-3)) \
            - n_leaf * lgamma(a) - (n_leaf-3) * lgamma(a * c)

        # uniform prior on topologies
        lnPrior -= torch.sum(torch.log(torch.arange(n_leaf*2-5, 0, -2)))

        return lnPrior
