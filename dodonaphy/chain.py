"""A Markov Chain"""
import os
import numpy as np
from dendropy.calculate.treecompare import weighted_robinson_foulds_distance as rf_dist
from dendropy.calculate.treecompare import symmetric_difference
from dendropy import Tree

from dodonaphy import tree as treeFunc
from dodonaphy import Chyp_np, Cphylo, Cpeeler, Cutils, peeler
from dodonaphy.base_model import BaseModel
from dodonaphy.Chyp_np import tangent_to_hyper as t02hyp
from dodonaphy.Chyp_np import tangent_to_hyper_jacobian as t02hyp_J


class Chain(BaseModel):
    """A Markov Chain class"""

    eps = 2.220446049250313e-16

    def __init__(
        self,
        partials,
        weights,
        dim,
        leaf_x=None,
        int_x=None,
        step_scale=0.01,
        chain_temp=1,
        target_acceptance=0.234,
        connector="nj",
        embedder="up",
        curvature=-1.0,
        converge_length=500,
        normalise_leaf=False,
        loss_fn="likelihood",
        matsumoto=False,
        tip_labels=None,
        warm_up=100,
        mcmc_alg="RAM",
    ):
        super().__init__(
            partials,
            weights,
            dim,
            soft_temp=None,
            embedder=embedder,
            connector=connector,
            curvature=curvature,
            normalise_leaf=normalise_leaf,
            loss_fn=loss_fn,
            require_grad=False,
            matsumoto=matsumoto,
            tip_labels=tip_labels,
        )
        self.leaf_x = leaf_x  # S x D
        self.int_x = int_x  # S-2 x D
        self.jacobian = np.zeros(1)
        if leaf_x is not None:
            self.S = leaf_x.shape()[0]
            self.X_bar = leaf_x.flatten()
        self.step_scale = step_scale
        self.cov = np.eye(self.S * self.D, dtype=np.double) * step_scale
        self.mcmc_alg = mcmc_alg
        if mcmc_alg == "AM" and warm_up <= 2:
            err_msg = f"Warm up period must be stricly greater than 2 for AM \
algorithm, got {warm_up}."
            raise ValueError(err_msg)
        self.warm_up = warm_up
        self.chain_temp = chain_temp
        self.accepted = 0
        self.iterations = 0
        self.target_acceptance = target_acceptance
        self.converge_length = converge_length
        if converge_length is not None:
            self.converged = [False] * converge_length
        self.more_tune = True

        self.ln_p = self.get_loss()
        self.ln_prior = Cphylo.compute_prior_gamma_dir_np(self.blens)
        self.rng = np.random.default_rng()

    def get_loss(self):
        """Get the current loss according to the objective.

        Returns:
            float: The loss value.
        """
        if self.loss_fn == "likelihood":
            self.ln_p = Cphylo.compute_LL_np(
                self.partials, self.weights, self.peel, self.blens
            )
        elif self.loss_fn == "pair_likelihood" and self.leaf_x is not None:
            pdm = Chyp_np.get_pdm(self.leaf_x, curvature=self.curvature)
            self.ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC" and self.leaf_x is not None:
            pdm = Chyp_np.get_pdm(self.leaf_x, curvature=self.curvature)
            self.ln_p = self.compute_hypHC(pdm, self.leaf_x)
        else:
            self.ln_p = -np.finfo(np.double).max
        return self.ln_p

    def set_probability(self):
        """Initialise likelihood and prior values of embedding"""
        if self.connector == "geodesics":
            self.peel, self.int_x = peeler.make_hard_peel_geodesic(self.leaf_x)
        elif self.connector == "nj":
            pdm = Chyp_np.get_pdm(self.leaf_x, curvature=self.curvature)
            self.peel, self.blens = Cpeeler.nj_np(pdm)

        if self.connector != "nj":
            self.blens = Cphylo.compute_branch_lengths_np(
                self.S,
                self.peel,
                self.leaf_x,
                self.int_x,
                curvature=self.curvature,
                matsumoto=self.matsumoto,
            )

        self.ln_p = self.get_loss()
        # self.ln_prior = self.compute_prior_birthdeath(self.peel, self.blens, **self.prior)
        self.ln_prior = Cphylo.compute_prior_gamma_dir_np(self.blens)

    def evolve(self):
        """Propose new embedding with regular MCMC."""
        proposal = self.sample_leaf_np(self.leaf_x, self.cov)
        ln_r_accept = self.ln_accept_ratio(proposal)

        self.check_proposal(proposal, ln_r_accept)

        if self.iterations == 0:
            self.X_bar = self.leaf_x.flatten()
        elif self.iterations <= self.warm_up:
            self.X_bar = self.X_bar + 1 / (self.iterations + 1) * (
                self.leaf_x.flatten() - self.X_bar
            )
        self.iterations += 1

    def ln_accept_ratio(self, prop):
        """Log acceptance critereon for Metropolis-Hastings

        Assumes a Hastings ratio of 1, i.e. symmetric proposals.

        Args:
            prop (dict): Proposal dictionary containing ln_p, ln_prior and the
            jacobian of the proposal.

        Returns:
            The acceptance ratio for MCMC.
        """
        ln_like_diff = prop["ln_p"] - self.ln_p
        ln_prior_diff = prop["ln_prior"] - self.ln_prior
        ln_jacob_diff = prop["jacobian"] - self.jacobian
        ln_hastings_diff = 0.0

        ln_r_accept = (
            ln_prior_diff + ln_like_diff + ln_jacob_diff
        ) * self.chain_temp + ln_hastings_diff
        return np.minimum(0.0, ln_r_accept)

    def euler_step(self, value, learn_rate=0.01):
        return self.step_scale + learn_rate * value

    def scale_step(self, sign, learn_rate=2.0):
        return np.power(learn_rate, sign) * self.step_scale

    def tune_step(self, decay=0.5):
        """Tune the acceptance rate.

        Use Euler method if acceptance rate is within 0.5 of target acceptance
        and is greater than 0.1. Solves:
            d(step)/d(acceptance) = acceptance - target_acceptance.
        Learning rate decays as (iteration + 1)^-decay when acceptance within
        0.1 of target.

        Otherwise scale the step by a factor of 10 (or 1/10 if step too big).


        Convergence is declared once the acceptance rate has been within tol
        of the target acceptance for self.converge_length consecutive iterations.

        Args:
            tol (float, optional): Tolerance. Defaults to 0.01.
        """
        if not self.more_tune or self.iterations == 0:
            return

        eta = (self.iterations + 1) ** -decay
        acceptance = self.accepted / self.iterations
        accept_diff = acceptance - self.target_acceptance
        if np.abs(acceptance - self.target_acceptance) < 0.1:
            new_step_scale = self.euler_step(accept_diff, learn_rate=eta)
        else:
            new_step_scale = self.scale_step(
                sign=accept_diff / np.abs(accept_diff), learn_rate=10.0
            )
        new_step_scale = np.maximum(new_step_scale, self.eps)
        self.cov = self.cov * new_step_scale / self.step_scale
        self.step_scale = new_step_scale
        self.check_convergence(accept_diff)

    def adapt_covariance(self):
        """Based on an Adaptive Metropolis Algoirthm, Haario 2001"""
        X = self.leaf_x.flatten()
        t = self.iterations
        s_d = 2.4 ** 2 / (self.S * self.D)
        X_bar_last = self.X_bar
        self.X_bar = X_bar_last + 1 / (t + 1) * (X - X_bar_last)

        self.cov = (t - 1) / t * self.cov + s_d / t * (
            t * np.outer(X_bar_last, X_bar_last)
            - (t - 1) * np.outer(self.X_bar, self.X_bar)
            + np.outer(X, X)
            + self.eps * np.eye(self.S * self.D)
        )
        return

    def check_proposal(self, proposal, ln_r_accept):
        accept = False
        if ln_r_accept >= 0:
            accept = True
        elif -np.random.exponential(scale=1.0) < ln_r_accept:
            accept = True
        if accept:
            self.leaf_x = proposal["leaf_x"]
            self.ln_p = proposal["ln_p"]
            self.ln_prior = proposal["ln_prior"]
            self.peel = proposal["peel"]
            self.blens = proposal["blens"]
            self.jacobian = proposal["jacobian"]
            if self.internals_exist:
                self.int_x = proposal["int_x"]
            self.accepted += 1
        return

    def evolve_RAM(self, path_write):
        """Based on Robust Adaptive Metropolis, Vihola 2012"""
        proposal = self.sample_leaf_np(self.leaf_x, self.cov)
        ln_r_accept = self.ln_accept_ratio(proposal)
        U = proposal["leaf_x"].flatten() - self.leaf_x.flatten()

        self.write_nni_dist(proposal, path_write)

        self.check_proposal(proposal, ln_r_accept)

        n = self.S * self.D
        eta = (self.iterations + 1) ** (-0.5)
        accept_diff = np.exp(ln_r_accept) - self.target_acceptance
        U_out = np.outer(U, U) / np.linalg.norm(U) ** 2
        cov_full = self.cov * (np.eye(n) + eta * accept_diff * U_out) * self.cov.T
        self.cov = np.linalg.cholesky(cov_full)
        self.iterations += 1
        return
    
    def write_nni_dist(self, proposal, path_write):
        if path_write is not None and self.chain_temp == 1:
            newick1 = treeFunc.tree_to_newick(self.tip_labels, self.peel, self.blens)
            tree1 = Tree.get(data=newick1, schema="newick")
            newick2 = treeFunc.tree_to_newick(self.tip_labels, proposal["peel"], proposal["blens"])
            tree2 = Tree.get(data=newick2, schema="newick", taxon_namespace=tree1.taxon_namespace)
            rfl = rf_dist(tree1, tree2)
            sym_diff = symmetric_difference(tree1, tree2)
            path_rfl = os.path.join(path_write, 'rfl.txt')
            path_sym_diff = os.path.join(path_write, 'sym_diff.txt')
            with open(path_rfl, "a", encoding="UTF-8") as file:
                file.write(f"{str(rfl)}\n")
            with open(path_sym_diff, "a", encoding="UTF-8") as file:
                file.write(f"{str(sym_diff)}\n")

    def check_convergence(self, accept_diff, tol=0.01):
        """Check for convergence of step.

        Previous 'converge_length' consecutive iterations must be within 'tol'
        of the target acceptance.

        Args:
            accept_diff ([type]): Difference to target acceptance.
            tol (float, optional): Absolute tolerance to target acceptance.
            Defaults to 0.01.
        """
        if self.converge_length is None:
            return
        self.converged.pop()
        self.converged.insert(0, np.abs(accept_diff) < tol)
        if all(self.converged):
            self.more_tune = False
            print(f"Step tuned to {self.step_scale}.")

    def sample_leaf_np(self, leaf_loc, leaf_cov):
        """Sample a nearby tree embedding.

        Each point is transformed R^n (using the self.embedding method), then
        a normal is sampled and transformed back to H^n. A tree is formed using
        the self.connect method.

        A dictionary is  returned containing information about this sampled tree.
        """
        leaf_x_prop, log_abs_det_jacobian = self.sample_loc_np(leaf_loc, leaf_cov)

        int_x_prop = None
        if self.connector == "nj":
            pdm = Chyp_np.get_pdm(leaf_x_prop, curvature=self.curvature)
            peel, blens = Cpeeler.nj_np(pdm)
        elif self.connector == "geodesics":
            peel, int_x_prop = peeler.make_hard_peel_geodesic(leaf_loc)
            blens = Cphylo.compute_branch_lengths_np(
                self.S,
                peel,
                leaf_x_prop,
                int_x_prop,
                self.curvature,
                matsumoto=self.matsumoto,
            )

        ln_p = Cphylo.compute_LL_np(self.partials, self.weights, peel, blens)
        ln_prior = Cphylo.compute_prior_gamma_dir_np(blens)

        proposal = {
            "leaf_x": leaf_x_prop,
            "peel": peel,
            "blens": blens,
            "jacobian": log_abs_det_jacobian,
            "ln_p": ln_p,
            "ln_prior": ln_prior,
        }
        if int_x_prop is not None:
            proposal["int_x"] = int_x_prop
        return proposal

    def sample_loc_np(self, loc_low, cov):
        """Sample points on hyperboloid.

        Points on the hyperboloid satisfy - x0^2 + x1^2 + ... + xdim^2= -1
        'low' locations are specified by the x1, ... xdim and x0 is determined.

        Args:
            loc_low (ndarray(double)): 'low' locations n_taxa x dim
            cov (ndarray(double)): Covariance matrix (n_taxa x dim) x (n_taxa x dim)

        Returns:
            [type]: [description]
        """
        n_locs = loc_low.shape[0]
        n_vars = n_locs * self.D
        loc_low_prop = np.zeros((n_locs, self.D))
        log_abs_det_jacobian = 0.0

        if self.embedder == "up":
            sample_hyp = self.rng.multivariate_normal(
                loc_low.flatten(), cov, method="cholesky"
            )
            loc_low_prop = sample_hyp.reshape((n_locs, self.D))

        elif self.embedder == "wrap":
            zero = np.zeros(n_vars, dtype=np.double)
            sample_t0 = self.rng.multivariate_normal(zero, cov, method="cholesky").reshape(
                (n_locs, self.D)
            )
            for i in range(n_locs):
                mu_hyp = Chyp_np.project_up(loc_low[i, :])
                loc_low_prop[i, :] = t02hyp(mu_hyp, sample_t0[i, :], self.D)[1:]
                log_abs_det_jacobian += t02hyp_J(mu_hyp, loc_low[i, :], self.D)

        if self.normalise_leaf:
            radius = np.linalg.norm(loc_low, axis=1)[0]
            loc_low_prop = Cutils.normalise_np(loc_low_prop) * radius

        return loc_low_prop, log_abs_det_jacobian


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
