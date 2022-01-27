"""A Markov Chain"""
import numpy as np

from . import Chyp_np, Cphylo, peeler
from .base_model import BaseModel


class Chain(BaseModel):
    """A Markov Chain class"""

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
        )
        self.leaf_x = leaf_x  # S x D
        self.int_x = int_x  # S-2 x D
        self.jacobian = np.zeros(1)
        if leaf_x is not None:
            self.S = leaf_x.shape()[0]
        self.step_scale = step_scale
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

    def get_loss(self):
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
        pdm = Chyp_np.get_pdm(self.leaf_x, curvature=self.curvature)
        if self.connector == "geodesics":
            self.peel, self.int_x = peeler.make_hard_peel_geodesic(self.leaf_x)
        elif self.connector == "nj":
            self.peel, self.blens = Cpeeler.nj_np(pdm)

        if self.connector != "nj":
            self.blens = Cphylo.compute_branch_lengths_np(
                self.S,
                self.peel,
                self.leaf_x,
                self.int_x,
                curvature=self.curvature,
            )

        self.ln_p = self.get_loss()
        # self.ln_prior = self.compute_prior_birthdeath(self.peel, self.blens, **self.prior)
        self.ln_prior = Cphylo.compute_prior_gamma_dir_np(self.blens)

    def evolve(self):
        """Propose new embedding"""
        proposal = self.sample_leaf_np(
            self.leaf_x,
            self.step_scale,
            self.connector,
            self.embedder,
            self.partials,
            self.weights,
            self.S,
            self.D,
            self.curvature,
            normalise_leaf=self.normalise_leaf,
        )

        r_accept = self.accept_ratio(proposal)

        accept = False
        if r_accept >= 1:
            accept = True

        elif np.random.uniform(low=0.0, high=1.0) < r_accept:
            accept = True

        if accept:
            self.leaf_x = proposal["leaf_x"]
            self.ln_p = proposal["ln_p"]
            self.ln_prior = proposal["ln_prior"]
            self.peel = proposal["peel"]
            self.blens = proposal["blens"]
            self.jacobian = proposal["jacobian"]
            if self.connector != "nj":
                self.int_x = proposal["int_x"]
            self.accepted += 1
        self.iterations += 1
        return accept

    def accept_ratio(self, prop):
        """Acceptance critereon for Metropolis-Hastings

        Args:
            prop ([type]): Proposal dictionary

        Returns:
            tuple: (r, prop_like)
            The acceptance ratio r and the likelihood of the proposal.
        """
        # likelihood ratio
        like_ratio = prop["ln_p"] - self.ln_p

        # prior ratio
        prior_ratio = prop["ln_prior"] - self.ln_prior

        # Jacobian ratio
        jacob_ratio = prop["jacobian"] - self.jacobian

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        # acceptance ratio
        r_accept = np.minimum(
            np.ones(1),
            np.exp(
                (prior_ratio + like_ratio + jacob_ratio) * self.chain_temp
                + hastings_ratio
            ),
        )

        return r_accept

    def euler_step(self, f, learn_rate=0.01):
        return self.step_scale + learn_rate * f

    def scale_step(self, sign, learn_rate=2.0):
        return np.power(learn_rate, sign) * self.step_scale

    def tune_step(self, tol=0.01):
        """Tune the acceptance rate.

        Use Euler method if acceptance rate is within 0.5 of target acceptance
        and is greater than 0.1. Solves:
            d(step)/d(acceptance) = acceptance - target_acceptance.
        Learning rate 0.01 and refined to 0.001 when acceptance within 0.1 of
        target.

        Otherwise scale the step by a factor of 10 (or 1/10 if step too big).


        Convergence is decalred once the acceptance rate has been within tol
        of the target acceptance for self.converge_length consecutive iterations.

        Args:
            tol (float, optional): Tolerance. Defaults to 0.01.
        """
        if not self.more_tune or self.iterations == 0:
            return

        acceptance = self.accepted / self.iterations
        accept_diff = acceptance - self.target_acceptance
        if np.abs(acceptance - self.target_acceptance) < 0.1:
            self.step_scale = self.euler_step(accept_diff, learn_rate=0.001)
        elif np.abs(acceptance - self.target_acceptance) < 0.5 and acceptance > 0.1:
            self.step_scale = self.euler_step(accept_diff, learn_rate=0.01)
        else:
            self.step_scale = self.scale_step(
                sign=accept_diff / np.abs(accept_diff), learn_rate=10.0
            )
        eps = 2.220446049250313e-16
        self.step_scale = np.maximum(self.step_scale, eps)
        # if np.isclose(self.step_scale, eps) :
            # declare stuck, reset to 1.0
            # self.step_scale = 1.0
        # print(f"step: {self.step_scale} acceptance:{acceptance}")

        # check convegence
        if self.converge_length is None:
            return
        self.converged.pop()
        self.converged.insert(0, np.abs(accept_diff) < tol)
        if all(self.converged):
            self.more_tune = False
            print(f"Step tuned to {self.step_scale}.")

    @staticmethod
    def sample_leaf_np(
        leaf_loc,
        leaf_cov_single,
        connector,
        embedder,
        partials,
        weights,
        taxa,
        dim,
        curvature,
        normalise_leaf=False,
    ):
        """Sample a nearby tree embedding.

        Each point is transformed R^n (using the self.embedding method), then
        a normal is sampled and transformed back to H^n. A tree is formed using
        the self.connect method.

        A dictionary is  returned containing information about this sampled tree.
        """
        leaf_cov = np.eye(taxa * dim, dtype=np.double) * leaf_cov_single

        leaf_x_prop, log_abs_det_jacobian = sample_loc_np(
            taxa,
            dim,
            leaf_loc,
            leaf_cov,
            embedder,
            is_internal=False,
            normalise_leaf=normalise_leaf,
        )

        int_x_prop = None
        if connector == "nj":
            pdm = Chyp_np.get_pdm(leaf_x_prop, curvature=curvature)
            peel, blens = peeler.nj_np(pdm)
        elif connector == "geodesics":
            peel, int_x_prop = peeler.make_hard_peel_geodesic(leaf_loc)
            blens = Cphylo.compute_branch_lengths_np(
                taxa,
                peel,
                leaf_x_prop,
                int_x_prop,
                curvature,
            )

        ln_p = Cphylo.compute_LL_np(partials, weights, peel, blens)
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

def sample_loc_np(n_taxa, dim, loc_hyp, cov, embedder, is_internal, normalise_leaf=False):
    """Sample points on hyperboloid."""
    if is_internal:
        n_locs = n_taxa - 2
    else:
        n_locs = n_taxa
    n_vars = n_locs * dim
    loc_hyp_prop = np.empty((n_taxa, dim+1))
    log_abs_det_jacobian = 0.0

    rng = np.random.default_rng()
    if embedder == "up":
        loc_t0 = loc_hyp[:, 1:]
        sample_t0 = rng.multivariate_normal(loc_t0.flatten(), cov)
        prop_t0 = sample_t0.reshape((n_locs, dim))
        for i in range(n_locs):
            loc_hyp_prop[i] = Chyp_np.project_up(prop_t0[i])

    elif embedder == "wrap":
        sample_t0 = rng.multivariate_normal(
            np.zeros(n_vars, dtype=np.double), cov
        )
        prop_t0 = sample_t0.reshape((n_locs, dim))
        for i in range(n_locs):
            loc_hyp_prop[i, :] = Chyp_np.tangent_to_hyper(loc_hyp[i, :], prop_t0[i, :], dim)
            log_abs_det_jacobian += Chyp_np.tangent_to_hyper_jacobian(loc_hyp_prop[i, :], loc_hyp[i, :-1], dim)

    if normalise_leaf:
        r = np.linalg.norm(loc_hyp_prop[0, 1:])
        loc_hyp_prop[:, 1:] = Cutils.normalise_np(loc_hyp_prop[:, 1:]) * r
        z = np.sqrt(np.sum(np.power(loc_hyp_prop, 2), 1) + 1)
        loc_hyp_prop[:, 0] = z

    return loc_hyp_prop, log_abs_det_jacobian


def normalise_LADJ(y):
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    n, D = y.shape

    log_abs_det_J = np.zeros(1)
    for k in range(n):
        J = np.linalg.det((np.eye(D, D) - np.outer(y[k], y[k]) / norm[k] ** 2) / norm[k])
        log_abs_det_J = log_abs_det_J + np.log(np.abs(J))
    return log_abs_det_J