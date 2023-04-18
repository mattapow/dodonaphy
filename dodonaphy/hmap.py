"""Maximum A Posteriori Module on hyperboloid sheet."""
import os
import time
import warnings

import numpy as np
import torch

from dodonaphy import Chyp_torch, peeler, tree, Cutils
from hydraPlus import hydraPlus
from dodonaphy.base_model import BaseModel


class HMAP(BaseModel):
    """Maximum A Posteriori class of embedding on hyperboloid sheet.

    Given a sequence alignment embed the tips onto the hyperboloid model of
    hyperbolic space, then optimise the tree likelihood of the decoded tree.

    """

    def __init__(
        self,
        partials,
        weights,
        dim,
        dists,
        soft_temp,
        loss_fn,
        path_write,
        curvature=-1.0,
        prior="None",
        tip_labels=None,
        matsumoto=False,
        connector="nj",
        peel=None,
        normalise_leaves=False,
        model_name="JC69"
    ):
        super().__init__(
            "hmap",
            partials,
            weights,
            dim=dim,
            soft_temp=soft_temp,
            connector=connector,
            curvature=curvature,
            loss_fn=loss_fn,
            tip_labels=tip_labels,
            model_name=model_name,
        )
        self.path_write = path_write
        self.normalise_leaves = normalise_leaves
        self.init_embedding_params(dists)
        self.init_model_params()
        self.current_epoch = 0
        self.prior = prior
        self.ln_p = self.compute_ln_likelihood()
        self.ln_prior = self.compute_ln_prior()
        self.matsumoto = matsumoto
        self.loss = torch.zeros(1)
        self.peel = peel

    def init_embedding_params(self, dists):
        # embed distances with hydra+
        hp_obj = hydraPlus.HydraPlus(dists, dim=self.D, curvature=np.array(self.curvature))
        emm_tips = hp_obj.embed(equi_adj=0.0, alpha=1.1)
        print("Embedding Strain (tips only) = {:.4}".format(emm_tips["stress_hydra"]))
        print(
            "Embedding Stress (tips only) = {:.4}".format(emm_tips["stress_hydraPlus"])
        )
        self.log(f"Embedding Strain (tips only) = {emm_tips['stress_hydra']}\n")
        self.log(f"Embedding Stress (tips only) = {emm_tips['stress_hydraPlus']}\n")
        # set locations as parameters to optimise
        if self.normalise_leaves:
            radius = np.mean(np.linalg.norm(emm_tips["X"], axis=1))
            directionals = Cutils.normalise_np(emm_tips["X"])
            self.params = {
                "radius": torch.tensor(radius, requires_grad=True, dtype=torch.float64),
                "directionals": torch.tensor(directionals, requires_grad=True, dtype=torch.float64)
            }
        else:
            self.params = {
                "leaf_loc": torch.tensor(
                    emm_tips["X"], requires_grad=True, dtype=torch.float64
                ),
            }

    def init_model_params(self):
        # set evolutionary model parameters to optimise
        if not self.phylomodel.fix_sub_rates:

            self.params["sub_rates"] = self.phylomodel.sub_rates
        if not self.phylomodel.fix_freqs:
            self.params["freqs"] = self.phylomodel.freqs

    def learn(self, epochs, learn_rate, save_locations, start=""):
        """Optimise params["dists"].

        NB: start is just a string for printing: which tree was used to
        generate the original distance matrix.

        """
        start_time = time.time()

        def lr_lambda(epoch):
            return 1.0 / (epoch + 1.0) ** 0.25

        # Consider using LBFGS, but appears to not perform as well.
        optimizer = torch.optim.Adam(params=list(self.params.values()), lr=learn_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #   optimizer, base_lr=learn_rate/100, max_lr=learn_rate, step_size_up=100)
        self.loss = -self.compute_ln_prior() - self.compute_ln_likelihood()
        post_hist = [-self.loss.item()]
        self.best_posterior = torch.tensor(-np.inf)

        if self.path_write is not None:
            self.save(epochs, learn_rate, start)
            self.record_if_best()

        def closure():
            optimizer.zero_grad()
            self.ln_prior = self.compute_ln_prior()
            self.ln_p = self.compute_ln_likelihood()
            self.loss = -self.ln_prior - self.ln_p
            self.record_if_best()
            self.loss.backward(retain_graph=True)
            return self.loss

        print(f"Running for {epochs} iterations.")
        print("Iteration: log prior + log_likelihood = log posterior")
        for i in range(epochs):
            self.current_epoch = i
            optimizer.step(closure)
            scheduler.step()
            post_hist.append(self.ln_p.item() + self.ln_prior.item())
            print(
                f"{i+1}: {self.ln_prior.item():.3f} + {self.ln_p.item():.3f} = {post_hist[-1]:.3f}"
            )
            if int(i + 1) % 10 == 9:
                print("")
            if self.path_write is not None:
                self.save_epoch(i, save_locations=save_locations)

        print(f"\nBest tree log posterior joint found: {self.best_posterior.item():.3f}")
        self.save_duration(start_time)
        self.save_best_tree()
        self.log(f"Best log likelihood: {self.best_ln_p}\n")
        self.log(f"Best log prior: {self.best_ln_prior.item()}\n")

        if epochs > 0 and self.path_write is not None:
            HMAP.trace(epochs + 1, post_hist, self.path_write, plot_hist=False)

    def save_best_tree(self):
        if self.path_write is not None:
            # TODO: this is the last model, on the best tree. Not the same.
            # save the last model
            file_model = os.path.join(self.path_write, f"{self.inference_name}_model.log")
            self.phylomodel.save(file_model)
            # save the best tree
            tree.save_tree_head(self.path_write, "mape", self.tip_labels)
            tree.save_tree(
                self.path_write,
                "mape",
                self.best_peel,
                self.best_blens,
                self.best_epoch,
                self.best_ln_p.item(),
                self.best_ln_prior.item(),
                self.name_id,
                last_tree=True,
            )

    def save_duration(self, start_time):
        if self.path_write is not None:
            seconds = time.time() - start_time
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            self.log(f"Total time: {int(hrs)}:{int(mins)}:{int(secs)}\n")

    def record_if_best(self):
        if self.loss < -self.best_posterior:
            self.best_posterior = -self.loss
            self.best_ln_p = self.ln_p
            self.best_ln_prior = self.ln_prior
            self.best_peel = self.peel
            self.best_blens = self.blens
            self.best_epoch = self.current_epoch

    def save(self, epochs, learn_rate, start):

        self.log("%-12s: %i\n" % ("# epochs", epochs))
        self.log("%-12s: %i\n" % ("Curvature", self.curvature))
        self.log("%-12s: %i\n" % ("Matsumoto", self.matsumoto))
        self.log("%s: %i\n" % ("Normalise Leaf", self.normalise_leaf))
        self.log("%-12s: %i\n" % ("Dimensions", self.D))
        self.log("%-12s: %i\n" % ("# Taxa", self.S))
        self.log("%-12s: %i\n" % ("# Patterns", self.L))
        self.log("%-12s: %f\n" % ("Learn Rate", learn_rate))
        self.log("%-12s: %f\n" % ("Soft temp", self.soft_temp))
        self.log("%-12s: %s\n" % ("Embed Mthd", self.embedder))
        self.log("%-12s: %s\n" % ("Connect Mthd", self.connector))
        self.log("%-12s: %s\n" % ("Loss function", self.loss_fn))
        self.log("%-12s: %s\n" % ("Prior", self.prior))
        self.log("%-12s: %s\n" % ("Start Tree", start))

    def save_epoch(self, i, save_locations=False):
        "Save posterior value and leaf locations to file."
        path_post = os.path.join(self.path_write, "posterior.txt")
        ln_p = self.ln_p.item()
        ln_prior = self.ln_prior.item()
        ln_post = ln_p + ln_prior
        if not os.path.isfile(path_post):
            with open(path_post, "w", encoding="UTF-8") as file:
                file.write("log prior, log likelihood, log posterior\n")
        with open(path_post, "a", encoding="UTF-8") as file:
            file.write(f"{ln_prior}, {ln_p}, {ln_post}\n")

        emm_path = os.path.join(self.path_write, "location")
        if save_locations:
            if not os.path.isdir(emm_path):
                os.mkdir(emm_path)
            emm_fn = os.path.join(emm_path, f"location_{i}.csv")
            print_header = ''.join([f"dim{i}, " for i in range(self.D)])
            locs = self.get_locs().detach().numpy()
            np.savetxt(
                emm_fn,
                locs,
                delimiter=", ",
                header=print_header,
            )
        tree.save_tree(
            self.path_write,
            "samples",
            self.peel,
            self.blens,
            i,
            self.ln_prior,
            self.ln_p,
            self.name_id
        )

    def get_locs(self):
        """Get current tip locations"""
        if self.normalise_leaves:
            locs = self.params["radius"] * self.params["directionals"]
        else:
            locs = self.params["leaf_loc"]
        return locs

    def connect(self, get_pdm=False):
        """Connect tips into a tree"""
        locs = self.get_locs()

        if self.connector == "geodesics":
            peel, _, blens = peeler.make_soft_peel_tips(
                locs, connector="geodesics", curvature=self.curvature
            )
            if get_pdm:
                pdm = Chyp_torch.get_pdm(locs, curvature=self.curvature)
        elif self.connector == "nj":
            pdm = Chyp_torch.get_pdm(locs, curvature=self.curvature)
            peel, blens = peeler.nj_torch(pdm, tau=self.soft_temp)
        elif self.connector == "fix":
            peel = self.peel
            pdm = Chyp_torch.get_pdm(locs, curvature=self.curvature)
            warnings.warn("NJ algorithm for branch lengths on fixed topology not guaranteed to work.")
            _, blens = peeler.nj_torch(pdm, tau=self.soft_temp, get_peel=False)
        else:
            raise ValueError(
                f"Connection must be one of 'nj', 'geodesics', 'fix'. Got {self.connector}"
            )
        if get_pdm:
            return peel, blens, pdm
        return peel, blens

    def compute_ln_likelihood(self):
        """Compute likelihood of current tree, reducing soft_temp as required."""
        if self.loss_fn == "likelihood":
            self.peel, self.blens = self.connect()
            self.ln_p = self.compute_LL(self.peel, self.blens, self.phylomodel.sub_rates, self.phylomodel.freqs)
            loss = self.ln_p
        elif self.loss_fn == "pair_likelihood":
            self.peel, self.blens, pdm = self.connect(get_pdm=True)
            self.ln_p = self.compute_LL(self.peel, self.blens, self.phylomodel.sub_rates, self.phylomodel.freqs)
            loss = self.compute_log_a_like(pdm, self.phylomodel.sub_rates, self.phylomodel.freqs)
        elif self.loss_fn == "hypHC":
            locs = self.get_locs()
            pdm = Chyp_torch.get_pdm(locs, curvature=self.curvature)
            loss = self.compute_likelihood_hypHC(
                pdm, locs, self.phylomodel.sub_rates, self.phylomodel.freqs, temperature=0.05, n_triplets=100)
        return loss

    def compute_ln_prior(self):
        if self.prior == "None":
            return torch.zeros(1)
        prior_sub_rates = self.phylomodel.compute_ln_prior_sub_rates(self.phylomodel.sub_rates)
        prior_freqs = self.phylomodel.compute_ln_prior_freqs(self.phylomodel.freqs)
        prior_tree = self.compute_ln_tree_prior()
        return prior_sub_rates + prior_freqs + prior_tree

    def compute_ln_tree_prior(self):
        """Compute prior of current tree."""
        if self.prior == "None":
            return torch.zeros(1)
        elif self.prior == "normal":
            locs = self.get_locs()
            return self.compute_prior_normal(locs)
        elif self.prior == "uniform":
            locs = self.get_locs()
            return self.compute_prior_unif(locs, scale=1.0)
        elif self.prior == "gammadir":
            return self.compute_prior_gamma_dir(self.blens)
        elif self.prior == "birthdeath":
            return self.compute_prior_birthdeath(self.peel, self.blens)
