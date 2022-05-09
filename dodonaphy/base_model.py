"Base model for MCMC and VI inference"
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from dendropy import Tree
from dendropy.model.birthdeath import birth_death_likelihood

from dodonaphy import poincare

from dodonaphy import Chyp_np, Chyp_torch
from dodonaphy import tree as treeFunc
from dodonaphy.phylo import JC69_p_t, calculate_treelikelihood
from dodonaphy.utils import LogDirPrior


class BaseModel(object):
    """Base Model for Inference"""

    def __init__(
        self,
        partials,
        weights,
        dim,
        soft_temp,
        curvature=-1.0,
        embedder="up",
        connector="nj",
        normalise_leaf=False,
        loss_fn="likelihood",
        require_grad=True,
        matsumoto=False,
        tip_labels=None,
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
        assert embedder in ("wrap", "up")
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
        self.loss_fn = loss_fn
        self.matsumoto = matsumoto
        if tip_labels is None:
            tip_labels = [f"T{i+1}" for i in range(self.S)]
        self.tip_labels = tip_labels

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
        leaf_x,
        int_x,
        curvature=-torch.ones(1),
        useNP=True,
        matsumoto=False,
    ):
        """Computes the hyperbolic distance of two points given in radial-\
            directional coordinates in the Poincare ball

        Args:
            n_taxa (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            leaf_x ([type]): [description]
            int_x ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Using numpy in cython may be faster
        if useNP:
            DTYPE = np.double
            leaf_x = leaf_x.detach().numpy().astype(DTYPE)
            int_x = int_x.detach().numpy().astype(DTYPE)
        bcount = 2 * n_taxa - 2
        blens = torch.empty(bcount, dtype=torch.float64)
        for b in range(n_taxa - 1):
            x2 = int_x[
                peel[b][2] - n_taxa - 1,
            ]

            for i in range(2):
                if peel[b][i] < n_taxa:
                    # leaf to internal
                    x1 = leaf_x[peel[b][i]]
                else:
                    # internal to internal
                    x1 = int_x[peel[b][i] - n_taxa - 1]

                if useNP:
                    hd = torch.tensor(Chyp_np.hyperbolic_distance(x1, x2, curvature))
                else:
                    hd = Chyp_torch.hyperbolic_distance(x1, x2, curvature)

                if matsumoto:
                    hd = torch.log(torch.cosh(hd))

                # add a tiny amount to avoid zero-length branches
                eps = torch.finfo(torch.double).eps
                blens[peel[b][i]] = torch.clamp(hd, min=eps)

        return blens

    def compute_LL(self, peel, blen):
        """Compute likelihood of tree.

        Args:
            peel ([type]): [description]
            blen ([type]): [description]

        Returns:
            [type]: [description]
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

    @staticmethod
    def compute_prior_birthdeath(peel, blen, **prior):
        """Calculates the log-likelihood of a tree under a birth death model.

        Args:
            peel
            blen
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
    def compute_prior_normal(locations, scale=0.1):
        """Multivariate Normal prior on locations.
        Centered at origin.

        Args:
            locations (ndarray or tensor): Locations n_points x n_dimensions
            scale (float, optional): Covariance scalar. Defaults to 0.01.

        Returns:
            _type_: Log probability of locations under Normal prior distribution.
        """
        np_flag = False
        if type(locations).__module__ == np.__name__:
            np_flag = True
            locations = torch.from_numpy(locations)
        if locations is None:
            return -np.infty

        n_taxa, dim = locations.size()
        cov = scale * torch.eye(n_taxa * dim, dtype=torch.double)
        mean = torch.zeros((n_taxa * dim), dtype=torch.double)
        prior_dist = MultivariateNormal(mean, covariance_matrix=cov)
        ln_prior = prior_dist.log_prob(locations.flatten())

        if np_flag:
            ln_prior = ln_prior.detach().numpy()
        return ln_prior


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

    @staticmethod
    def trace(epochs, like_hist, path_write):
        """Plot trace and histogram of likelihood."""
        plt.figure()
        plt.plot(range(epochs), like_hist, "r", label="likelihood")
        plt.xlabel("Epochs")
        plt.ylabel("likelihood")
        plt.legend()
        plt.savefig(path_write + "/likelihood_trace.png")

        plt.clf()
        plt.hist(like_hist)
        plt.title("Likelihood histogram")
        plt.savefig(path_write + "/likelihood_hist.png")

def normalise_LADJ(loc):
    """Return the log of the absolute value of the determinant of the jacobian.
    Normalising points to unit sphere.

    Args:
        loc (ndarray): locations to normalise: n_locations x n_dim

    Returns:
        float: log(|det(Jacobian)|)
    """
    norm = np.linalg.norm(loc, axis=-1, keepdims=True)
    n_loc, dim = loc.shape

    log_abs_det_j = 0.0
    for k in range(n_loc):
        j_det = np.linalg.det(
            (np.eye(dim, dim) - np.outer(loc[k], loc[k]) / norm[k] ** 2) / norm[k]
        )
        log_abs_det_j = log_abs_det_j + np.log(np.abs(j_det))
    return log_abs_det_j
