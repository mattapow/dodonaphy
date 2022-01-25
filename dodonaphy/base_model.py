"Base model for MCMC and VI inference"
import math

import numpy as np
import torch
from dendropy import Tree
from dendropy.model.birthdeath import birth_death_likelihood as birth_death_likelihood
from torch.distributions.multivariate_normal import MultivariateNormal

from dodonaphy import poincare

from . import Chyperboloid, Chyperboloid_np, peeler, utils, Cphylo
from . import tree as treeFunc
from .phylo import JC69_p_t, calculate_treelikelihood
from .utils import LogDirPrior


class BaseModel(object):
    """Base Model for Inference"""

    def __init__(
        self,
        partials,
        weights,
        dim,
        soft_temp,
        curvature=-1.0,
        embedder="simple",
        connector="nj",
        normalise_leaf=False,
        loss_fn="likelihood",
        require_grad=True,
    ):
        self.partials = partials.copy()
        self.weights = weights
        self.S = len(self.partials)
        self.L = self.partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        self.soft_temp = soft_temp
        assert curvature <= 0
        if require_grad:
            self.curvature = torch.tensor(curvature)
        else:
            self.curvature = curvature
        self.epoch = 0
        assert embedder in ("wrap", "simple")
        self.embedder = embedder
        assert connector in ("geodesics", "nj")
        self.connector = connector
        self.internals_exist = False
        self.peel = np.zeros((self.S - 1, 3), dtype=int)
        if require_grad:
            self.blens = torch.zeros(self.bcount, dtype=torch.double)
        else:
            self.blens = np.zeros(self.bcount, dtype=np.double)
        self.normalise_leaf = normalise_leaf
        assert loss_fn in ("likelihood", "pair_likelihood", "hypHC")
        self.loss_fn = loss_fn

        # make space for internal partials
        for _ in range(self.S - 1):
            if require_grad:
                self.partials.append(
                    torch.zeros(
                        (1, 4, self.L), dtype=torch.float64, requires_grad=False
                    )
                )
            else:
                self.partials.append(
                    torch.zeros(
                        (1, 4, self.L), dtype=torch.float64, requires_grad=False
                    )
                )

    @staticmethod
    def compute_branch_lengths(
        n_taxa,
        peel,
        leaf_r,
        leaf_dir,
        int_r,
        int_dir,
        curvature=-torch.ones(1),
        useNP=True,
    ):
        """Computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball

        Args:
            n_taxa (integer): [description]
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
        bcount = 2 * n_taxa - 2
        blens = torch.empty(bcount, dtype=torch.float64)
        for b in range(n_taxa - 1):
            directional2 = int_dir[
                peel[b][2] - n_taxa - 1,
            ]
            r2 = int_r[peel[b][2] - n_taxa - 1]

            for i in range(2):
                if peel[b][i] < n_taxa:
                    # leaf to internal
                    r1 = leaf_r[peel[b][i]]
                    directional1 = leaf_dir[peel[b][i], :]
                else:
                    # internal to internal
                    r1 = int_r[peel[b][i] - n_taxa - 1]
                    directional1 = int_dir[
                        peel[b][i] - n_taxa - 1,
                    ]

                if useNP:
                    hd = torch.tensor(
                        Chyperboloid_np.hyperbolic_distance(
                            r1, r2, directional1, directional2, curvature
                        )
                    )
                else:
                    hd = Chyperboloid.hyperbolic_distance(
                        r1, r2, directional1, directional2, curvature
                    )

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
        return calculate_treelikelihood(
            self.partials,
            self.weights,
            peel,
            mats,
            torch.full([4], 0.25, dtype=torch.float64),
        )

    def compute_log_a_like(self, pdm):
        """Compute the log-a-like function of the embedding.

        The log-probability of all the pairwise taxa.
        """
        eps = torch.finfo(torch.double).eps
        P = torch.zeros((4, 4, self.L))

        for i in range(self.S):
            mats = JC69_p_t(pdm[i])
            for j in range(i - 1):
                P = P + torch.log(
                    torch.clamp(torch.matmul(mats[j], self.partials[i]), min=eps)
                )

        L = torch.sum(self.weights)
        return torch.sum(P * self.weights) / L

    def compute_likelihood_hypHC(
        self, dists_data, leaf_X, temperature=0.05, n_triplets=100
    ):
        eps = torch.finfo(torch.double).eps
        likelihood_dist = torch.zeros_like(dists_data)
        for i in range(self.S):
            mats = JC69_p_t(dists_data[i])
            for j in range(i - 1):
                P_ij = torch.log(
                    torch.clamp(torch.matmul(mats[j], self.partials[i]), min=eps)
                )
                dist_ij = -torch.sum(P_ij * self.weights)
                likelihood_dist[i, j] = dist_ij
                likelihood_dist[j, i] = dist_ij

        return self.compute_hypHC(
            likelihood_dist, leaf_X, temperature=temperature, n_triplets=n_triplets
        )

    def compute_hypHC(self, dists_data, leaf_X, temperature=0.05, n_triplets=100):
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
                self.get_similarities(
                    triplets[i, [0, 0, 1]], triplets[i, [1, 2, 2]], dists_data
                )
            )

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

    def get_similarities(self, u, v, pdm_data):
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


    def sample(
        self,
        leaf_loc,
        leaf_cov,
        int_loc=None,
        int_cov=None,
        soft=True,
        normalise_leaf=False,
    ):
        """Sample a nearby tree embedding.

        Each point is transformed R^n (using the self.embedding method), then
        a normal is sampled and transformed back to H^n. A tree is formed using
        the self.connect method.

        A dictionary is  returned containing information about this sampled tree.
        """
        # reshape covariance if single number
        if torch.numel(leaf_cov) == 1:
            leaf_cov = torch.eye(self.S * self.D, dtype=torch.double) * leaf_cov
        if int_cov is not None and torch.numel(int_cov) == 1:
            int_cov = torch.eye((self.S - 2) * self.D, dtype=torch.double) * int_cov

        leaf_r_prop, leaf_dir_prop, log_abs_det_jacobian, log_Q = self.sample_loc(
            leaf_loc, leaf_cov, is_internal=False, normalise_leaf=normalise_leaf
        )

        if self.internals_exist:
            (
                int_r_prop,
                int_dir_prop,
                log_abs_det_jacobian_int,
                log_Q_int,
            ) = self.sample_loc(
                int_loc, int_cov, is_internal=True, normalise_leaf=False
            )
            min_leaf_r = min(leaf_r_prop)
            int_r_prop[int_r_prop > min_leaf_r] = min_leaf_r
            log_abs_det_jacobian = log_abs_det_jacobian + log_abs_det_jacobian_int
            log_Q = log_Q + log_Q_int

        # internal nodes and peel for geodesics
        if self.connector == "geodesics":
            leaf_locs = leaf_r_prop.repeat((self.D, 1)).T * leaf_dir_prop
            if isinstance(leaf_locs, torch.Tensor):
                peel, int_locs, blens = peeler.make_soft_peel_tips(
                    leaf_locs, connector="geodesics", curvature=self.curvature
                )
            else:
                peel, int_locs = peeler.make_hard_peel_geodesic(leaf_locs)
            int_r_prop, int_dir_prop = utils.cart_to_dir(int_locs)

        # get peels
        if self.connector == "nj":
            pdm = Chyperboloid.get_pdm_torch(
                leaf_r_prop, leaf_dir_prop, curvature=self.curvature
            )
            if soft:
                peel, blens = peeler.nj(pdm, tau=self.soft_temp)
            else:
                peel, blens = peeler.nj(pdm)
        elif self.connector == "mst":
            peel = peeler.make_peel_mst(
                leaf_r_prop, leaf_dir_prop, int_r_prop, int_dir_prop
            )
        elif self.connector == "mst_choice":
            peel = self.select_peel_mst(
                leaf_r_prop, leaf_dir_prop, int_r_prop, int_dir_prop
            )

        # get proposal branch lengths
        if self.connector != "nj":
            blens = self.compute_branch_lengths(
                self.S,
                peel,
                leaf_r_prop,
                leaf_dir_prop,
                int_r_prop,
                int_dir_prop,
                useNP=False,
            )

        # get log likelihood
        if self.loss_fn == "likelihood":
            ln_p = self.compute_LL(peel, blens)
        elif self.loss_fn == "pair_likelihood":
            ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC":
            leaf_X = utils.dir_to_cart(leaf_r_prop, leaf_dir_prop)
            ln_p = self.compute_hypHC(pdm, leaf_X)

        # get log prior
        ln_prior = self.compute_prior_gamma_dir(blens)

        if self.connector in ("nj"):
            proposal = {
                "leaf_r": leaf_r_prop,
                "leaf_dir": leaf_dir_prop,
                "peel": peel,
                "blens": blens,
                "jacobian": log_abs_det_jacobian,
                "logQ": log_Q,
                "ln_p": ln_p,
                "ln_prior": ln_prior,
            }
        elif self.connector in ("geodesics", "mst", "mst_choice"):
            proposal = {
                "leaf_r": leaf_r_prop,
                "leaf_dir": leaf_dir_prop,
                "int_r": int_r_prop,
                "int_dir": int_dir_prop,
                "peel": peel,
                "blens": blens,
                "jacobian": log_abs_det_jacobian,
                "logQ": log_Q,
                "ln_p": ln_p,
                "ln_prior": ln_prior,
            }
        return proposal

    def sample_loc(self, loc, cov, is_internal, normalise_leaf=False):
        """Given locations in poincare ball, transform them to Euclidean
        space, sample from a Normal and transform sample back."""
        if is_internal:
            n_locs = self.S - 2
        else:
            n_locs = self.S
        n_vars = n_locs * self.D
        if self.embedder == "simple":
            # transform internals to R^n
            loc_t0 = Chyperboloid.ball2real(loc)
            log_abs_det_jacobian = -Chyperboloid.real2ball_LADJ(loc_t0)

            # flatten data to sample
            loc_t0 = loc_t0.reshape(n_vars)

            # propose new int nodes from normal in R^n
            normal_dist = MultivariateNormal(loc_t0.squeeze(), cov)
            sample = normal_dist.rsample()
            log_Q = normal_dist.log_prob(sample)
            loc_t0 = sample.reshape((n_locs, self.D))

            # convert ints to poincare ball
            loc_prop = Chyperboloid.real2ball(loc_t0)

        elif self.embedder == "wrap":
            # transform ints to R^n
            loc_t0, jacobian = Chyperboloid.p2t0(loc, get_jacobian=True)
            loc_t0 = loc_t0.clone()
            log_abs_det_jacobian = -jacobian

            # propose new int nodes from normal in R^n
            normal_dist = MultivariateNormal(
                torch.zeros(n_vars, dtype=torch.double), cov
            )
            sample = normal_dist.rsample()
            log_Q = normal_dist.log_prob(sample)
            loc_prop = Chyperboloid.t02p(
                sample.reshape(n_locs, self.D),
                loc_t0.reshape(n_locs, self.D),
            )

        if normalise_leaf:
            # TODO: do we need normalise jacobian? The positions are inside the integral... so yes
            r_prop = torch.norm(loc_prop[0, :]).repeat(self.S)
            loc_prop = utils.normalise(loc_prop) * r_prop.repeat((self.D, 1)).T
        else:
            r_prop = torch.norm(loc_prop, dim=-1)
        dir_prop = loc_prop / torch.norm(loc_prop, dim=-1, keepdim=True)
        return r_prop, dir_prop, log_abs_det_jacobian, log_Q

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

        birth_rate = prior.get("birth_rate", 2.0)
        death_rate = prior.get("death_rate", 0.5)

        tipnames = ["T" + str(x + 1) for x in range(S)]
        newick = treeFunc.tree_to_newick(tipnames, peel, blen)
        tree = Tree.get(data=newick, schema="newick")
        LL = birth_death_likelihood(
            tree=tree,
            ultrametricity_precision=False,
            birth_rate=birth_rate,
            death_rate=death_rate,
        )
        return torch.tensor(LL)

    @staticmethod
    def compute_prior_gamma_dir(
        blen,
        aT=torch.ones(1),
        bT=torch.full((1,), 0.1),
        a=torch.ones(1),
        c=torch.ones(1),
    ):
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
        ln_prior = LogDirPrior(blen, aT, bT, a, c)

        # with prefactor
        lgamma = torch.lgamma
        ln_prior = (
            ln_prior
            + (aT) * torch.log(bT)
            - lgamma(aT)
            + lgamma(a * n_leaf + a * c * (n_leaf - 3))
            - n_leaf * lgamma(a)
            - (n_leaf - 3) * lgamma(a * c)
        )

        # uniform prior on topologies
        ln_prior = ln_prior - torch.sum(torch.log(torch.arange(n_leaf * 2 - 5, 0, -2)))

        return ln_prior
