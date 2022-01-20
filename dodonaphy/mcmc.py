"""Markov Chain Monte Calo Module"""
import os
import time

import numpy as np

from . import hydra, peeler, tree, utils
from . import Cutils, Cpeeler, Cphylo, Cmcmc, Chyperboloid_np
from .base_model import BaseModel


class Chain(BaseModel):
    """A Markov Chain"""

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
        connector="mst",
        embedder="simple",
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
        elif self.connector == "mst":
            self.peel = peeler.make_peel_mst(
                self.leaf_r, self.leaf_dir, self.int_r, self.int_dir
            )
        elif self.connector == "mst_choice":
            self.peel = self.select_peel_mst(
                self.leaf_r, self.leaf_dir, self.int_r, self.int_dir
            )

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
        leaf_loc = self.leaf_dir * np.tile(self.leaf_r, (self.D, 1)).T
        if self.connector == "mst":
            int_loc = self.int_dir * np.tile(self.int_r, (2, 1)).T
            proposal = self.sample(
                leaf_loc,
                self.step_scale,
                int_loc,
                self.step_scale,
                soft=False,
                normalise_leaf=self.normalise_leaf,
            )
        else:
            proposal = self.sample_leaf_np(
                leaf_loc,
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
            self.step_scale = self.scale_step(sign=accept_diff / np.abs(accept_diff), learn_rate=10.0)
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
        # TODO: allow other loss functions
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


class DodonaphyMCMC:
    """Markov Chain Monte Carlo"""

    def __init__(
        self,
        partials,
        weights,
        dim,
        connector="mst",
        embedder="simple",
        step_scale=0.01,
        n_chains=1,
        curvature=-1.0,
        save_period=1,
        n_grids=10,
        n_trials=10,
        max_scale=1,
        normalise_leaf=False,
        loss_fn="likelihood",
    ):
        self.n_chains = n_chains
        self.chain = []
        d_temp = 0.1
        self.n_grids = n_grids
        self.n_trials = n_trials
        self.max_scale = max_scale
        self.save_period = save_period
        for i in range(n_chains):
            chain_temp = 1.0 / (1 + d_temp * i)
            self.chain.append(
                Chain(
                    partials,
                    weights,
                    dim,
                    step_scale=step_scale,
                    chain_temp=chain_temp,
                    embedder=embedder,
                    connector=connector,
                    curvature=curvature,
                    normalise_leaf=normalise_leaf,
                    loss_fn=loss_fn,
                    converge_length=None,
                )
            )

    def run_burnin(self, burnin):
        """Run burn in iterations without saving."""
        print(f"Burning in for {burnin} iterations.")
        deceile = 1
        for i in range(burnin):
            if i / burnin * 10 > deceile:
                print(f"{deceile * 10:d}%% ", end="", flush=True)
                deceile += 1
            for chain in self.chain:
                chain.evolve()
                chain.tune_step()
            if self.n_chains > 1:
                _ = self.swap()
        print("100%")

    def print_iter(self, iteration):
        """Print current state."""
        if iteration > 0 and (iteration / self.save_period) % 10 == 0:
            print("")
        print(
            f"{iteration:9d} --- {self.chain[0].ln_p:.3f}",
            end="",
        )
        if self.n_chains > 1:
            print(" (", end="")
            for chain in self.chain[1:]:
                print(
                    f" {chain.ln_p:.3f}",
                    end="",
                )
            print(")", end="")
        if iteration > 0:
            print(" --- ", end="")
            for chain in self.chain:
                print(
                    f" {chain.accepted / chain.iterations:5.3f}, ",
                    end="",
                    flush=True,
                )
        print("")

    def learn(self, epochs, burnin=0, path_write="./out"):
        """Run the markov chains."""
        print(f"Using 1 cold chain and {int(self.n_chains - 1)} hot chains.")

        start = time.time()
        if path_write is not None:
            info_file = path_write + "/" + "mcmc.info"
            self.save_info(info_file, epochs, burnin, self.save_period)
            tree.save_tree_head(path_write, "mcmc", self.chain[0].S)

        for chain in self.chain:
            chain.set_probability()

        if burnin > 0:
            self.run_burnin(burnin)

        swaps = 0
        print(f"Running for {epochs} iterations.\n")
        print("Iteration --- Log Likelihood (hot chains) --- Acceptance Rates")
        self.print_iter(0)
        if path_write is not None:
            self.save_iteration(path_write, 0)
        for epoch in range(1, epochs + 1):
            for chain in self.chain:
                chain.evolve()
                chain.tune_step()

            do_save = self.save_period > 0 and epoch % self.save_period == 0
            if do_save:
                self.print_iter(epoch)
                if path_write is not None:
                    self.save_iteration(path_write, epoch)

            if self.n_chains > 1:
                swaps += self.swap()

        if path_write is not None:
            self.save_final_info(path_write, swaps, time.time() - start)

    def save_info(self, file, epochs, burnin, save_period):
        """Save information about this simulation."""
        with open(file, "w", encoding="UTF-8") as file:
            file.write(f"# epochs:  {epochs}\n")
            file.write(f"Burnin: {burnin}\n")
            file.write(f"Save period: {save_period}\n")
            file.write(f"Dimensions: {self.chain[0].D}\n")
            file.write(f"# Taxa:  {self.chain[0].S}\n")
            file.write(f"Unique sites:  {self.chain[0].L}\n")
            file.write(f"\nChains:  {self.n_chains}\n")
            for chain in self.chain:
                file.write(f"\nChain temp:  {chain.chain_temp}\n")
                file.write(f"Convergence length: {chain.converge_length}\n")
                file.write(f"Connect Mthd:  {chain.connector}\n")
                file.write(f"Embed Mthd:  {chain.embedder}\n")

    def save_final_info(self, path_write, swaps, seconds):
        file_name = path_write + "/" + "mcmc.info"
        with open(file_name, "a", encoding="UTF-8") as file:
            for c_id, chain in enumerate(self.chain):
                final_accept = chain.accepted / chain.iterations
                file.write(f"Chain {c_id} acceptance: {final_accept}\n")
                file.write(f"Step Scale tuned to:  {chain.step_scale}\n\n")
                if chain.more_tune and chain.converge_length is not None:
                    file.write(f"Chain {c_id} did not converge to target acceptance.\n")
            file.write(f"\nTotal chain swaps: {swaps}\n")
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            file.write(f"Total time: {int(hrs)}:{int(mins)}:{int(secs)}\n")

    def save_iteration(self, path_write, iteration):
        """Save the current state to file."""
        if self.chain[0].loss_fn != "likelihood":
            ln_p = Cphylo.compute_LL_np(
                self.chain[0].partials,
                self.chain[0].weights,
                self.chain[0].peel,
                self.chain[0].blens,
            )
        else:
            ln_p = self.chain[0].ln_p
        tree.save_tree(
            path_write,
            "mcmc",
            self.chain[0].peel,
            self.chain[0].blens,
            iteration,
            float(ln_p),
            float(self.chain[0].ln_prior),
        )
        file_name = path_write + "/locations.csv"
        if not os.path.isfile(file_name):
            with open(file_name, "a", encoding="UTF-8") as file:
                for i in range(len(self.chain[0].leaf_r)):
                    file.write(f"leaf_{i}_r, ")
                for i in range(len(self.chain[0].leaf_dir)):
                    for j in range(self.chain[0].D):
                        file.write(f"leaf_{i}_dir_{j}, ")

                if self.chain[0].internals_exist:
                    for i in range(len(self.chain[0].int_r)):
                        file.write(f"int_{i}_r, ")
                    for i in range(len(self.chain[0].int_dir)):
                        for j in range(self.chain[0].D):
                            file.write(f"int_{i}_dir_{j}")
                            if not (
                                j == self.chain[0].D - 1
                                and i == len(self.chain[0].int_dir)
                            ):
                                file.write(", ")
                file.write("\n")

        with open(file_name, "a", encoding="UTF-8") as file:
            file.write(
                np.array2string(self.chain[0].leaf_r, separator=",")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )
            file.write(", ")
            file.write(
                np.array2string(self.chain[0].leaf_dir, separator=",")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )

            if self.chain[0].internals_exist:
                file.write(", ")
                file.write(
                    np.array2string(self.chain[0].int_r.data.numpy(), separator=", ")
                    .replace("\n", "")
                    .replace("[", "")
                    .replace("]", "")
                )
                file.write(",")
                file.write(
                    np.array2string(self.chain[0].int_dir.data.numpy(), separator=", ")
                    .replace("\n", "")
                    .replace("[", "")
                    .replace("]", "")
                )
            file.write("\n")

    def swap(self):
        """randomly swap states in 2 chains according to MCMCMC"""

        # Pick two adjacent chains
        rng = np.random.default_rng()
        i = rng.multinomial(1, np.ones(self.n_chains - 1) / (self.n_chains - 1))[0] - 1
        j = i + 1
        chain_i = self.chain[i]
        chain_j = self.chain[j]

        # get log posterior (unnormalised)
        ln_post_i = chain_i.ln_p + chain_i.ln_prior
        ln_post_j = chain_j.ln_p + chain_j.ln_prior

        # probability of exhanging these two chains
        prob1 = (ln_post_i - ln_post_j) * chain_j.chain_temp
        prob2 = (ln_post_j - ln_post_i) * chain_i.chain_temp
        r_accept = np.minimum(1, np.exp(prob1 + prob2))

        # swap with probability r
        if r_accept > np.random.uniform(low=0.0, high=1.0):
            # swap the locations and current probability
            chain_i.leaf_r, chain_j.leaf_r = (
                chain_j.leaf_r,
                chain_i.leaf_r,
            )
            chain_i.leaf_dir, chain_j.leaf_dir = (
                chain_j.leaf_dir,
                chain_i.leaf_dir,
            )
            chain_i.int_r, chain_j.int_r = (
                chain_j.int_r,
                chain_i.int_r,
            )
            chain_i.int_dir, chain_j.int_dir = (
                chain_j.int_dir,
                chain_i.int_dir,
            )
            chain_i.ln_p, chain_j.ln_p = (
                chain_j.ln_p,
                chain_i.ln_p,
            )
            chain_i.ln_prior, chain_j.ln_prior = (
                chain_j.ln_prior,
                chain_i.ln_prior,
            )
            return 1
        return 0

    def initialise_chains(self, emm, normalise=True):
        """initialise each chain"""
        for chain in self.chain:
            if normalise:
                chain.leaf_r = emm["r"]
            else:
                chain.leaf_r = np.mean(emm["r"]).repeat(self.chain[0].S)
            chain.leaf_dir = emm["directional"]
            chain.n_points = len(chain.leaf_dir)
            chain.int_r = None
            chain.int_dir = None

            if chain.connector in ("mst", "mst_choice"):
                int_r, int_dir = chain.initialise_ints(
                    emm,
                    n_grids=self.n_grids,
                    n_trials=self.n_trials,
                    max_scale=self.max_scale,
                )
                chain.int_r = int_r.astype(np.double)
                chain.int_dir = int_dir.astype(np.double)

    @staticmethod
    def run(
        dim,
        partials,
        weights,
        dists_data,
        path_write=None,
        epochs=1000,
        step_scale=0.01,
        save_period=1,
        burnin=0,
        n_grids=10,
        n_trials=10,
        max_scale=1,
        n_chains=1,
        connector="mst",
        embedder="simple",
        curvature=-1.0,
        normalise_leaf=True,
        loss_fn="likelihood",
    ):
        """Run Dodonaphy's MCMC."""
        print("\nRunning Dodonaphy MCMC")
        assert connector in ["mst", "geodesics", "nj", "mst_choice"]

        # embed tips with distances using Hydra
        emm_tips = hydra.hydra(
            dists_data, dim=dim, curvature=curvature, stress=True, equi_adj=0.0
        )
        print(f"Embedding Stress (tips only) = {emm_tips['stress'].item():.4}")

        mymod = DodonaphyMCMC(
            partials,
            weights,
            dim,
            step_scale=step_scale,
            n_chains=n_chains,
            connector=connector,
            embedder=embedder,
            curvature=curvature,
            save_period=save_period,
            n_grids=n_grids,
            n_trials=n_trials,
            max_scale=max_scale,
            normalise_leaf=normalise_leaf,
            loss_fn=loss_fn,
        )

        mymod.initialise_chains(emm_tips, normalise=normalise_leaf)
        mymod.learn(epochs, burnin=burnin, path_write=path_write)
