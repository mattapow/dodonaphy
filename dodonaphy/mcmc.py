import os

import numpy as np
import torch
from torch.distributions.uniform import Uniform

from . import Cutils, hydra, peeler, tree, utils
from .base_model import BaseModel


class Chain(BaseModel):
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
        curvature=-1,
    ):
        super().__init__(
            partials,
            weights,
            dim,
            soft_temp=None,
            embedder=embedder,
            connector=connector,
            curvature=curvature,
        )
        self.leaf_dir = leaf_dir  # S x D
        self.int_dir = int_dir  # S-2 x D
        self.int_r = int_r  # S-2
        self.leaf_r = leaf_r  # single scalar
        self.jacobian = torch.zeros(1)
        if leaf_dir is not None:
            self.S = len(leaf_dir)
        self.step_scale = torch.tensor(step_scale, requires_grad=True)
        self.chain_temp = chain_temp
        self.accepted = 0
        self.iterations = 0
        self.target_acceptance = target_acceptance
        self.converged = [False] * 200
        self.moreTune = True
        self.ln_p = self.compute_LL(self.peel, self.blens)
        self.ln_prior = self.compute_prior_gamma_dir(self.blens)

    def set_probability(self):
        # initialise likelihood and prior values of embedding

        # set peel + poincare locations
        pdm = Cutils.get_pdm_torch(
            self.leaf_r.repeat(self.S), self.leaf_dir, curvature=self.curvature
        )
        if self.connector in ("geodesics", "incentre"):
            loc_poin = self.leaf_dir * self.leaf_r
            self.peel, int_locs = peeler.make_peel_tips(loc_poin, self.connector)
            self.int_r, self.int_dir = utils.cart_to_dir(int_locs)
            leaf_r_all, self.leaf_dir = utils.cart_to_dir(loc_poin)
            self.leaf_r = leaf_r_all[0]
        elif self.connector == "nj":
            self.peel, self.blens = peeler.nj(pdm)
        elif self.connector == "mst":
            self.peel = peeler.make_peel_mst(
                self.leaf_r.repeat(self.S), self.leaf_dir, self.int_r, self.int_dir
            )
        elif self.connector == "mst_choice":
            self.peel = self.select_peel_mst(
                self.leaf_r.repeat(self.S), self.leaf_dir, self.int_r, self.int_dir
            )

        # set blens
        if self.connector != "nj":
            self.blens = self.compute_branch_lengths(
                self.S,
                self.peel,
                self.leaf_r.repeat(self.S),
                self.leaf_dir,
                self.int_r,
                self.int_dir,
            )

        # current likelihood
        # self.ln_p = self.compute_log_a_like(pdm)
        self.ln_p = self.compute_LL(self.peel, self.blens)
        # leaf_X = utils.dir_to_cart(self.leaf_r, self.leaf_dir)
        # self.ln_p = self.compute_hypHC(leaf_X)

        # current prior
        # self.ln_prior = self.compute_prior_birthdeath(self.peel, self.blens, **self.prior)
        self.ln_prior = self.compute_prior_gamma_dir(self.blens)

    def evolve(self):
        # propose new embedding
        leaf_loc = self.leaf_r * self.leaf_dir
        if self.connector == "mst":
            int_loc = self.int_dir * torch.tile(self.int_r, (2, 1)).transpose(
                dim0=0, dim1=1
            )
            proposal = self.sample(
                leaf_loc, self.step_scale, int_loc, self.step_scale, soft=False
            )
        else:
            proposal = self.sample(leaf_loc, self.step_scale, soft=False)

        # Decide whether to accept proposal
        r = self.accept_ratio(proposal)

        accept = False
        if r >= 1:
            accept = True
        elif Uniform(torch.zeros(1), torch.ones(1)).sample() < r:
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

    def accept_ratio(self, p):
        """Acceptance critereon for Metropolis-Hastings

        Args:
            p ([type]): Proposal dictionary

        Returns:
            tuple: (r, prop_like)
            The acceptance ratio r and the likelihood of the proposal.
        """
        # likelihood ratio
        like_ratio = p["ln_p"] - self.ln_p

        # prior ratio
        prior_ratio = p["ln_prior"] - self.ln_prior

        # Jacobian ratio
        jacob_ratio = p["jacobian"] - self.jacobian

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        # acceptance ratio
        r = torch.minimum(
            torch.ones(1),
            torch.exp(
                (prior_ratio + like_ratio + jacob_ratio) * self.chain_temp
                + hastings_ratio
            ),
        )

        return r

    def tune_step(self, tol=0.01):
        """Tune the acceptance rate. Simple Euler method.

        Args:
            tol (float, optional): Tolerance. Defaults to 0.01.
        """
        if not self.moreTune or self.iterations == 0:
            return

        lr = torch.tensor(0.001)
        eps = torch.tensor(torch.finfo(torch.double).eps)
        acceptance = self.accepted / self.iterations
        dy = acceptance - self.target_acceptance
        # print(f"new step={self.step_scale + lr * dy}")
        self.step_scale = torch.maximum(self.step_scale + lr * dy, eps)

        # check convegence
        self.converged.pop()
        self.converged.insert(0, np.abs(dy) < tol)
        if all(self.converged):
            self.moreTune = False
            print("Step tuned to %f." % self.step_scale)


class DodonaphyMCMC:
    def __init__(
        self,
        partials,
        weights,
        dim,
        connector="mst",
        embedder="simple",
        step_scale=0.01,
        nChains=1,
        curvature=-1.0,
        save_period=1,
    ):
        self.nChains = nChains
        self.chain = []
        dTemp = 0.1
        self.save_period = save_period
        for i in range(nChains):
            chain_temp = 1.0 / (1 + dTemp * i)
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
                )
            )

    def learn(self, epochs, burnin=0, path_write="./out"):
        print("Using 1 cold chain and %d hot chains." % int(self.nChains - 1))

        if path_write is not None:
            info_file = path_write + "/" + "mcmc.info"
            self.save_info(info_file, epochs, burnin, self.save_period)
            tree.save_tree_head(path_write, "mcmc", self.chain[0].S)

        # Initialise prior and likelihood
        for c in range(self.nChains):
            self.chain[c].set_probability()

        if burnin > 0:
            print("Burning in for %d iterations." % burnin)
            deceile = 1
            for i in range(burnin):
                if i / burnin * 10 > deceile:
                    print("%d%% " % (deceile * 10), end="", flush=True)
                    deceile += 1
                for c in range(self.nChains):
                    # step
                    self.chain[c].evolve()
                    # tune step
                    self.chain[c].tune_step()

                # swap 2 chains
                if self.nChains > 1:
                    _ = self.swap()
            print("100%")

        swaps = 0
        print("Running for %d iterations.\n" % epochs)
        for i in range(epochs):
            for c in range(self.nChains):
                # step
                self.chain[c].evolve()
                # tune step if not converged
                self.chain[c].tune_step()

            # save
            doSave = self.save_period > 0 and i % self.save_period == 0
            if doSave:
                if path_write is not None:
                    self.save_iteration(path_write, i)

                if i > 0:
                    print(
                        "Iteration: %i LnL: %f Acceptance Rate: %5.3f"
                        % (
                            i,
                            self.chain[0].ln_p,
                            self.chain[0].accepted / self.chain[0].iterations,
                        ),
                        end="",
                        flush=True,
                    )

                    if self.nChains > 1:
                        print(" (", end="")
                        for c in range(self.nChains - 1):
                            print(
                                " %5.3f"
                                % (
                                    self.chain[c + 1].accepted
                                    / self.chain[c + 1].iterations
                                ),
                                end="",
                            )
                        print(")")
                    else:
                        print("")

            # swap 2 chains
            if self.nChains > 1:
                swaps += self.swap()

        if path_write is not None:
            fn = path_write + "/" + "mcmc.info"
            with open(fn, "a", encoding='UTF-8') as file:
                final_accept = np.average(
                    [
                        self.chain[c].accepted / self.chain[c].iterations
                        for c in range(self.nChains)
                    ]
                )
                file.write("%-12s: %f\n" % ("Acceptance", final_accept))
                file.write("%-12s: %d\n" % ("Swaps", swaps))

    def save_info(self, file, epochs, burnin, save_period):
        with open(file, "w", encoding='UTF-8') as f:
            f.write("%-12s: %i\n" % ("# epochs", epochs))
            f.write("%-12s: %i\n" % ("Burnin", burnin))
            f.write("%-12s: %i\n" % ("Save period", save_period))
            f.write("%-12s: %i\n" % ("Dimensions", self.chain[0].D))
            f.write("%-12s: %i\n" % ("# Taxa", self.chain[0].S))
            f.write("%-12s: %i\n" % ("Unique sites", self.chain[0].L))
            f.write("%-12s: %i\n" % ("Chains", self.nChains))
            for i in range(self.nChains):
                f.write("%-12s: %f\n" % ("Chain temp", self.chain[i].chain_temp))
                f.write("%-12s: %f\n" % ("Step Scale", self.chain[i].step_scale))
                f.write("%-12s: %s\n" % ("Connect Mthd", self.chain[i].connector))
                f.write("%-12s: %s\n" % ("Embed Mthd", self.chain[i].embedder))

    def save_iteration(self, path_write, iteration):
        ln_p = self.chain[0].compute_LL(self.chain[0].peel, self.chain[0].blens)
        tree.save_tree(
            path_write,
            "mcmc",
            self.chain[0].peel,
            self.chain[0].blens,
            iteration,
            float(ln_p),
            float(self.chain[0].ln_prior),
        )
        fn = path_write + "/locations.csv"
        if not os.path.isfile(fn):
            with open(fn, "a", encoding='UTF-8') as file:
                file.write("leaf_r, ")
                for i in range(len(self.chain[0].leaf_dir)):
                    for j in range(self.chain[0].D):
                        file.write("leaf_%d_dir_%d, " % (i, j))
                if self.chain[0].int_r is not None:
                    for i in range(len(self.chain[0].int_r)):
                        file.write("int_%d_r, " % (i))
                    for i in range(len(self.chain[0].int_dir)):
                        for j in range(self.chain[0].D):
                            file.write("int_%d_dir_%d" % (i, j))
                            if not (
                                j == self.chain[0].D - 1
                                and i == len(self.chain[0].int_dir)
                            ):
                                file.write(", ")
                file.write("\n")

        with open(fn, "a", encoding='UTF-8') as file:
            file.write(
                np.array2string(self.chain[0].leaf_r.data.numpy(), separator=", ")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )
            file.write(", ")
            file.write(
                np.array2string(self.chain[0].leaf_dir.data.numpy(), separator=", ")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )

            if self.chain[0].connector == "mst":
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
        """
        randomly swap states in 2 chains according to MCMCMC
        """

        # Pick two adjacent chains
        i = torch.multinomial(torch.ones(self.nChains - 1), 1, replacement=False)
        j = i + 1

        # get log posterior (unnormalised)
        ln_post_i = self.chain[i].ln_p + self.chain[i].ln_prior
        ln_post_j = self.chain[j].ln_p + self.chain[j].ln_prior

        # probability of exhanging these two chains
        prob1 = (ln_post_i - ln_post_j) * self.chain[j].chain_temp
        prob2 = (ln_post_j - ln_post_i) * self.chain[i].chain_temp
        r = torch.minimum(torch.ones(1), torch.exp(prob1 + prob2))

        # swap with probability r
        if r > Uniform(torch.zeros(1), torch.ones(1)).rsample():
            # swap the locations and current probability
            self.chain[i].leaf_r, self.chain[j].leaf_r = (
                self.chain[j].leaf_r,
                self.chain[i].leaf_r,
            )
            self.chain[i].leaf_dir, self.chain[j].leaf_dir = (
                self.chain[j].leaf_dir,
                self.chain[i].leaf_dir,
            )
            self.chain[i].int_r, self.chain[j].int_r = (
                self.chain[j].int_r,
                self.chain[i].int_r,
            )
            self.chain[i].int_dir, self.chain[j].int_dir = (
                self.chain[j].int_dir,
                self.chain[i].int_dir,
            )
            self.chain[i].ln_p, self.chain[j].ln_p = self.chain[j].ln_p, self.chain[i].ln_p
            self.chain[i].ln_prior, self.chain[j].ln_prior = (
                self.chain[j].ln_prior,
                self.chain[i].ln_prior,
            )
            return 1
        return 0

    def initialise_chains(self, emm, n_grids=10, n_trials=10, max_scale=2):
        # initialise each chain
        for i in range(self.nChains):
            # put leaves on a sphere
            self.chain[i].leaf_r = torch.tensor(np.mean(emm["r"], dtype=np.double))
            self.chain[i].leaf_dir = torch.from_numpy(
                emm["directional"].astype(np.double)
            )
            self.chain[i].n_points = len(self.chain[i].leaf_dir)
            self.chain[i].int_r = None
            self.chain[i].int_dir = None

            if self.chain[i].connector in ("mst", "mst_choice"):
                int_r, int_dir = self.chain[i].initialise_ints(
                    emm, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale
                )
                self.chain[i].int_r = torch.from_numpy(int_r.astype(np.double))
                self.chain[i].int_dir = torch.from_numpy(int_dir.astype(np.double))

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
        n_trials=100,
        max_scale=1,
        nChains=1,
        connector="mst",
        embedder="simple",
        curvature=-1.0,
    ):
        print("\nRunning Dodonaphy MCMC")
        assert connector in ["incentre", "mst", "geodesics", "nj", "mst_choice"]

        # embed tips with distances using Hydra
        emm_tips = hydra.hydra(
            dists_data, dim=dim, curvature=curvature, stress=True, equi_adj=0.0
        )
        print("Embedding Stress (tips only) = {:.4}".format(emm_tips["stress"].item()))

        with torch.no_grad():
            # Initialise model
            mymod = DodonaphyMCMC(
                partials,
                weights,
                dim,
                step_scale=step_scale,
                nChains=nChains,
                connector=connector,
                embedder=embedder,
                curvature=curvature,
                save_period=save_period,
            )

            # Choose internal node locations from best random initialisation
            mymod.initialise_chains(
                emm_tips, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale
            )

            # Learn
            mymod.learn(epochs, burnin=burnin, path_write=path_write)
