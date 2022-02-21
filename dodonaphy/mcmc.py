"""Markov Chain Monte Calo Module"""
import os
import time

import numpy as np
from . import tree, Cphylo, hydraPlus
from .chain import Chain


class DodonaphyMCMC:
    """Markov Chain Monte Carlo"""

    def __init__(
        self,
        partials,
        weights,
        dim,
        connector="nj",
        embedder="up",
        step_scale=0.01,
        n_chains=1,
        curvature=-1.0,
        save_period=1,
        n_grids=10,
        n_trials=10,
        max_scale=1,
        normalise_leaf=False,
        loss_fn="likelihood",
        swap_period=1000,
        n_swaps=10,
        matsumoto=False,
        tip_labels=None,
    ):
        self.n_chains = n_chains
        self.chains = []
        d_temp = 0.1
        self.n_grids = n_grids
        self.n_trials = n_trials
        self.max_scale = max_scale
        self.save_period = save_period
        self.swap_period = swap_period
        self.n_swaps = n_swaps
        for i in range(n_chains):
            chain_temp = 1.0 / (1 + d_temp * i)
            self.chains.append(
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
                    matsumoto=matsumoto,
                    tip_labels=tip_labels,
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
            for chain in self.chains:
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
            f"{iteration:9d} --- {self.chains[0].ln_p:.3f}",
            end="",
        )
        if self.n_chains > 1:
            print(" (", end="")
            for chain in self.chains[1:]:
                print(
                    f" {chain.ln_p:.3f}",
                    end="",
                )
            print(")", end="")
        if iteration > 0:
            print(" --- ", end="")
            for chain in self.chains:
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
            tree.save_tree_head(path_write, "mcmc", self.chains[0].tip_labels)

        for chain in self.chains:
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
            for chain in self.chains:
                chain.evolve()
                chain.tune_step()

            do_save = self.save_period > 0 and epoch % self.save_period == 0
            if do_save:
                self.print_iter(epoch)
                if path_write is not None:
                    self.save_iteration(path_write, epoch)

            try_swap = self.n_chains > 1 and epoch % self.swap_period == self.swap_period - 1
            if try_swap:
                for _ in range(self.n_swaps):
                    swaps += self.swap()

        if path_write is not None:
            self.save_final_info(path_write, swaps, time.time() - start)

    def save_info(self, file, epochs, burnin, save_period):
        """Save information about this simulation."""
        with open(file, "w", encoding="UTF-8") as file:
            file.write(f"# epochs:  {epochs}\n")
            file.write(f"Burnin: {burnin}\n")
            file.write(f"Save period: {save_period}\n")
            file.write(f"Dimensions: {self.chains[0].D}\n")
            file.write(f"# Taxa:  {self.chains[0].S}\n")
            file.write(f"Unique sites:  {self.chains[0].L}\n")
            file.write(f"\nChains:  {self.n_chains}\n")
            for chain in self.chains:
                file.write(f"\nChain temp:  {chain.chain_temp}\n")
                file.write(f"Convergence length: {chain.converge_length}\n")
                file.write(f"Connect Mthd:  {chain.connector}\n")
                file.write(f"Embed Mthd:  {chain.embedder}\n")

    def save_final_info(self, path_write, swaps, seconds):
        """Save tail of info file with acceptance and time taken."""
        file_name = path_write + "/" + "mcmc.info"
        with open(file_name, "a", encoding="UTF-8") as file:
            for c_id, chain in enumerate(self.chains):
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
        if self.chains[0].loss_fn != "likelihood":
            ln_p = Cphylo.compute_LL_np(
                self.chains[0].partials,
                self.chains[0].weights,
                self.chains[0].peel,
                self.chains[0].blens,
            )
        else:
            ln_p = self.chains[0].ln_p
        tree.save_tree(
            path_write,
            "mcmc",
            self.chains[0].peel,
            self.chains[0].blens,
            iteration,
            float(ln_p),
            float(self.chains[0].ln_prior),
        )
        file_name = path_write + "/locations.csv"
        if not os.path.isfile(file_name):
            with open(file_name, "a", encoding="UTF-8") as file:
                for i in range(len(self.chains[0].leaf_x)):
                    for j in range(self.chains[0].D):
                        file.write(f"leaf_{i}_x_{j}, ")

                if self.chains[0].internals_exist:
                    for i in range(len(self.chains[0].int_x)):
                        for j in range(self.chains[0].D):
                            file.write(f"int_{i}_x_{j}")
                            if not (
                                j == self.chains[0].D - 1
                                and i == len(self.chains[0].int_x)
                            ):
                                file.write(", ")
                file.write("\n")

        with open(file_name, "a", encoding="UTF-8") as file:
            file.write(
                np.array2string(self.chains[0].leaf_x, separator=",")
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
            )

            if self.chains[0].internals_exist:
                file.write(", ")
                file.write(
                    np.array2string(self.chains[0].int_x.data.numpy(), separator=", ")
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
        chain_i = self.chains[i]
        chain_j = self.chains[j]

        # get log posterior (unnormalised)
        ln_post_i = chain_i.ln_p + chain_i.ln_prior
        ln_post_j = chain_j.ln_p + chain_j.ln_prior

        # probability of exhanging these two chains
        prob1 = (ln_post_i - ln_post_j) * chain_j.chain_temp
        prob2 = (ln_post_j - ln_post_i) * chain_i.chain_temp
        ln_r_accept = np.minimum(0, prob1 + prob2)

        # swap with probability r
        if ln_r_accept > -np.random.exponential(scale=1.0):
            # swap the locations and current probability
            chain_i.leaf_x, chain_j.leaf_x = (
                chain_j.leaf_x,
                chain_i.leaf_x,
            )
            chain_i.int_x, chain_j.int_x = (
                chain_j.int_x,
                chain_i.int_x,
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

    def initialise_chains(self, emm):
        """initialise each chain"""
        for chain in self.chains:
            chain.leaf_x = emm["X"]
            chain.n_points = chain.leaf_x.shape[0]

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
        n_chains=1,
        connector="nj",
        embedder="up",
        curvature=-1.0,
        normalise_leaf=True,
        loss_fn="likelihood",
        swap_period=1000,
        n_swaps=10,
        matsumoto=False,
        tip_labels=None,
    ):
        """Run Dodonaphy's MCMC."""
        print("\nRunning Dodonaphy MCMC")
        assert connector in ["geodesics", "nj"]

        # embed tips with distances using HydraPlus
        hp_obj = hydraPlus.HydraPlus(dists_data, dim=dim, curvature=curvature)
        emm_tips = hp_obj.embed(equi_adj=0.0, stress=True)
        print(f"Embedding stress of tips (hydra+) = {emm_tips['stress_hydraPlus']:.4}")

        partials = [partial.detach().numpy() for partial in partials]
        weights = weights.detach().numpy()
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
            normalise_leaf=normalise_leaf,
            loss_fn=loss_fn,
            swap_period=swap_period,
            n_swaps=n_swaps,
            matsumoto=matsumoto,
            tip_labels=tip_labels,
        )

        mymod.initialise_chains(emm_tips)
        mymod.learn(epochs, burnin=burnin, path_write=path_write)
