"""A Markov Chain"""
import numpy as np

from . import Chyperboloid_np, Cmcmc, Cpeeler, Cphylo, Cutils, peeler, utils
from .base_model import BaseModel


class Chain(BaseModel):
    """A Markov Chain class"""

    def __init__(
        self,
        partials,
        weights,
        dim,
        leaf_r=None,
        leaf_dir=None,
        int_r=None,
        int_dir=None,
        step_scale=0.01,
        chain_temp=1,
        target_acceptance=0.234,
        embedder="simple",
        connector="nj",
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
        self.leaf_dir = leaf_dir  # S x D
        self.int_dir = int_dir  # S-2 x D
        self.int_r = int_r  # S-2
        self.leaf_r = leaf_r  # S
        self.jacobian = np.zeros(1)
        if leaf_dir is not None:
            self.S = len(leaf_dir)
        self.step_scale = step_scale
        self.chain_temp = chain_temp
        self.accepted = 0
        self.iterations = 0
        self.target_acceptance = target_acceptance
        self.converge_length = converge_length
        if converge_length is not None:
            self.converged = [False] * converge_length
        self.more_tune = True

        if self.loss_fn == "likelihood":
            self.ln_p = Cphylo.compute_LL_np(
                self.partials, self.weights, self.peel, self.blens
            )
        elif self.loss_fn == "pair_likelihood" and self.leaf_r is not None:
            pdm = Chyperboloid_np.get_pdm(
                self.leaf_r, self.leaf_dir, curvature=self.curvature, dtype="numpy"
            )
            self.ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC" and self.leaf_r is not None:
            pdm = Chyperboloid_np.get_pdm(
                self.leaf_r, self.leaf_dir, curvature=self.curvature, dtype="numpy"
            )
            leaf_X = utils.dir_to_cart(self.leaf_r, self.leaf_dir)
            self.ln_p = self.compute_hypHC(pdm, leaf_X)
        else:
            self.ln_p = -np.finfo(np.double).max
        self.ln_prior = Cphylo.compute_prior_gamma_dir_np(self.blens)

    def set_probability(self):
        """Initialise likelihood and prior values of embedding"""
        pdm = Chyperboloid_np.get_pdm_tips_np(
            self.leaf_r, self.leaf_dir, curvature=self.curvature
        )
        if self.connector == "geodesics":
            loc_poin = self.leaf_dir * np.tile(self.leaf_r, (self.D, 1)).T
            self.peel, int_locs = peeler.make_hard_peel_geodesic(loc_poin)
            self.int_r, self.int_dir = Cutils.cart_to_dir_np(int_locs)
            self.leaf_r, self.leaf_dir = Cutils.cart_to_dir_np(loc_poin)
        elif self.connector == "nj":
            self.peel, self.blens = Cpeeler.nj_np(pdm)

        if self.connector != "nj":
            self.blens = Cphylo.compute_branch_lengths_np(
                self.S,
                self.peel,
                self.leaf_r,
                self.leaf_dir,
                self.int_r,
                self.int_dir,
                curvature=self.curvature,
            )

        # current likelihood
        if self.loss_fn == "likelihood":
            self.ln_p = Cphylo.compute_LL_np(
                self.partials, self.weights, self.peel, self.blens
            )
        elif self.loss_fn == "pair_likelihood":
            self.ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC":
            leaf_X = utils.dir_to_cart(self.leaf_r, self.leaf_dir)
            self.ln_p = self.compute_hypHC(pdm, leaf_X)

        # current prior
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
            self.leaf_r = proposal["leaf_r"]
            self.leaf_dir = proposal["leaf_dir"]
            self.ln_p = proposal["ln_p"]
            self.ln_prior = proposal["ln_prior"]
            self.peel = proposal["peel"]
            self.blens = proposal["blens"]
            self.jacobian = proposal["jacobian"]
            if self.connector != "nj":
                self.int_r = proposal["int_r"]
                self.int_dir = proposal["int_dir"]
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
        self.step_scale = np.maximum(self.step_scale, 2.220446049250313e-16)

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

        leaf_r_prop, leaf_dir_prop, log_abs_det_jacobian = Cmcmc.sample_loc_np(
            taxa,
            dim,
            leaf_loc,
            leaf_cov,
            embedder,
            is_internal=False,
            normalise_leaf=normalise_leaf,
        )

        if connector == "nj":
            pdm = Chyperboloid_np.get_pdm_tips_np(
                leaf_r_prop, leaf_dir_prop, curvature=curvature
            )
            peel, blens = Cpeeler.nj_np(pdm)
        elif connector == "geodesics":
            leaf_locs = np.tile(leaf_r_prop, (dim, 1)).T * leaf_dir_prop
            peel, int_locs = peeler.make_hard_peel_geodesic(leaf_locs)
            int_r_prop, int_dir_prop = Cutils.cart_to_dir_np(int_locs)
            blens = Cphylo.compute_branch_lengths_np(
                taxa,
                peel,
                leaf_r_prop,
                leaf_dir_prop,
                int_r_prop,
                int_dir_prop,
                curvature,
            )

        # get log likelihood
        # if self.loss_fn == "likelihood":
        ln_p = Cphylo.compute_LL_np(partials, weights, peel, blens)
        # elif self.loss_fn == "pair_likelihood":
        #     ln_p = self.compute_log_a_like(pdm)
        # elif self.loss_fn == "hypHC":
        #     leaf_X = Cutils.dir_to_cart(leaf_r_prop, leaf_dir_prop)
        #     ln_p = self.compute_hypHC(pdm, leaf_X)

        # get log prior
        ln_prior = Cphylo.compute_prior_gamma_dir_np(blens)

        if connector in ("nj"):
            proposal = {
                "leaf_r": leaf_r_prop,
                "leaf_dir": leaf_dir_prop,
                "peel": peel,
                "blens": blens,
                "jacobian": log_abs_det_jacobian,
                "ln_p": ln_p,
                "ln_prior": ln_prior,
            }
        elif connector in ("geodesics"):
            proposal = {
                "leaf_r": leaf_r_prop,
                "leaf_dir": leaf_dir_prop,
                "int_r": int_r_prop,
                "int_dir": int_dir_prop,
                "peel": peel,
                "blens": blens,
                "jacobian": log_abs_det_jacobian,
                "ln_p": ln_p,
                "ln_prior": ln_prior,
            }
        return proposal
