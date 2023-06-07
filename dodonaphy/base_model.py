"Base model for MCMC and VI inference"
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from dendropy import Tree
from dendropy.model.birthdeath import birth_death_likelihood

from dodonaphy import poincare

from dodonaphy import Chyp_np, Chyp_torch, phylo
from dodonaphy import tree as treeFunc
from dodonaphy.phylo import calculate_treelikelihood
from dodonaphy.utils import LogDirPrior
from dodonaphy.phylomodel import PhyloModel

import importlib

bito_spec = importlib.util.find_spec("bito")
bito_found = bito_spec is not None
if bito_found:
    import bito


class BaseModel(object):
    """Base Model for Inference of trees on the Hyperboloid."""

    def __init__(
        self,
        inference_name,
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
        model_name="JC69",
        freqs=None,
    ):
        self.inference_name = inference_name
        self.partials = partials.copy()
        self.weights = weights
        self.S = len(self.partials)
        self.L = self.partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        self.soft_temp = soft_temp
        self.require_grad = require_grad
        self.curvature = curvature
        self.epoch = 0
        assert embedder in ("wrap", "up")
        self.embedder = embedder
        assert connector in ("geodesics", "nj", "nj-r", "fix")
        self.connector = connector
        if self.connector == "fix":
            self.internals_exist = True
        else:
            self.internals_exist = False
        self.peel = np.zeros((self.S - 1, 3), dtype=int)
        if self.require_grad:
            self.blens = torch.zeros(self.bcount, dtype=torch.double)
        else:
            self.blens = np.zeros(self.bcount, dtype=np.double)
        self.normalise_leaf = normalise_leaf
        self.loss_fn = loss_fn
        self.loss = torch.zeros([1])
        self.matsumoto = matsumoto
        if tip_labels is None:
            tip_labels = [f"T{i+1}" for i in range(self.S)]
        self.tip_labels = tip_labels
        self.name_id = {name: id for id, name in enumerate(self.tip_labels)}

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
        if not require_grad:
            self.partials = [partial.detach().numpy() for partial in self.partials]
        # phylogenetic model
        self.use_bito = False
        self.phylomodel = PhyloModel(model_name)
        if freqs is not None:
            self.phylomodel.freqs = torch.from_numpy(freqs)

    @property
    def curvature(self):
        if self.require_grad:
            return -torch.exp(self._curvature)
        else:
            return -np.exp(self._curvature)

    @curvature.setter
    def curvature(self, curvature):
        if self.require_grad:
            curvature = torch.tensor(curvature)
            if not hasattr(self, "_curvature"):
                self._curvature = torch.log(-curvature).clone().detach().requires_grad_(True)
            else:
                self._curvature = torch.log(-curvature)
        else:
            self._curvature = np.log(-curvature)

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

    def log(self, message):
        if self.path_write is not None:
            file_name = os.path.join(self.path_write, f"{self.inference_name}.log")
            with open(file_name, "a", encoding="UTF-8") as file:
                file.write(message)

    def init_model_params(self):
        # set evolutionary model parameters to optimise
        if not self.phylomodel.fix_sub_rates:
            self.params_optim["sub_rates"] = self.phylomodel._sub_rates
        if not self.phylomodel.fix_freqs:
            self.params_optim["freqs"] = self.phylomodel._freqs
        if self.require_grad:
            self.params_optim["curvature"] = self._curvature

    def init_bito(self, msa_file, peel):
        self.use_bito = True
        self.bito_inst = bito.unrooted_instance("dodonaphy")
        self.bito_inst.read_fasta_file(str(msa_file))  # read alignment
        self.model_specification = bito.PhyloModelSpecification(
            substitution=self.phylomodel.name, site="constant", clock="strict"
        )
        parent_id = phylo.get_parent_id_vector(peel, rooted=False)
        tree = bito.UnrootedTree.of_parent_id_vector(parent_id)
        self.bito_inst.tree_collection = bito.UnrootedTreeCollection([tree], self.tip_labels)
        self.bito_inst.prepare_for_phylo_likelihood(self.model_specification, 1)

    def compute_LL(self, peel, blen):
        """Compute likelihood of tree.

        Args:
            peel ([type]): [description]
            blen ([type]): [description]
            sub_rates ([type]): [description]
            freqs ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.use_bito:
            return phylo.TreeLikelihood.apply(
                blen,
                peel,
                self.bito_inst,
                self.phylomodel._sub_rates,
                self.phylomodel._freqs,
            )
        else:
            mats = self.phylomodel.get_transition_mats(blen)
            return calculate_treelikelihood(
                self.partials,
                self.weights,
                peel,
                mats,
                self.phylomodel.freqs,
            )

    def compute_log_a_like(self, pdm):
        """Compute the log-a-like function of the embedding.

        The log-probability of all the pairwise taxa.
        """
        eps = torch.finfo(torch.double).eps
        P = torch.zeros((4, 4, self.L))

        for i in range(self.S):
            mats = self.phylomodel.get_transition_mats(torch.clamp(pdm[i], min=eps))
            for j in range(self.S):
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
            mats = self.phylomodel.get_transition_mats(torch.clamp(dists_data[i], min=eps))
            for j in range(self.S):
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
            similarities[i, :] = self.get_similarities(
                triplets[i, [0, 0, 1]], triplets[i, [1, 2, 2]], dists_data
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
    def compute_prior_birthdeath(peel, blen, tipnames, **prior):
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
    def compute_prior_normal(locations, scale=0.01):
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
    def compute_prior_unif(locations, scale=1.0):
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

        _scale = torch.tensor(scale)
        prior_dist = Uniform(-_scale, _scale)
        ln_prior = torch.sum(prior_dist.log_prob(locations.flatten()))

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
        ln_dir = LogDirPrior(blen, aT, bT, a, c)

        # with prefactor
        lgamma = torch.lgamma
        ln_prefactor = (
            (aT) * torch.log(bT)
            - lgamma(aT)
            + lgamma(a * n_leaf + a * c * (n_leaf - 3))
            - n_leaf * lgamma(a)
            - (n_leaf - 3) * lgamma(a * c)
        )

        # uniform prior on topologies
        if n_leaf <= 2:
            ln_topo = 0.0
        else:
            ln_topo = torch.sum(torch.log(torch.arange(n_leaf * 2 - 5, 0, -2)))

        ln_prior = ln_dir + ln_prefactor - ln_topo

        return ln_prior

    @staticmethod
    def trace(epochs, like_hist, path_write, plot_hist=True):
        """Plot trace and histogram of likelihood."""
        plt.figure()
        plt.plot(range(epochs), like_hist, "r", label="likelihood")
        plt.xlabel("Epochs")
        plt.ylabel("likelihood")
        plt.legend()
        plt.savefig(path_write + "/likelihood_trace.png")

        if plot_hist:
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
