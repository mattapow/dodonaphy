"""Maximum A Posteriori Module on hyperboloid sheet."""
import os

import numpy as np
import torch

from dodonaphy import Chyp_torch, peeler, tree
import hydraPlus
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
        curvature=-1.0,
        prior="None",
        tip_labels=None,
        matsumoto=False,
    ):
        super().__init__(
            partials,
            weights,
            dim=dim,
            soft_temp=soft_temp,
            connector="nj",
            curvature=curvature,
            loss_fn=loss_fn,
            tip_labels=tip_labels,
        )
        hp_obj = hydraPlus.HydraPlus(dists, dim=self.D, curvature=curvature)
        emm_tips = hp_obj.embed(equi_adj=0.0, stress=True)

        self.params = {
            "leaf_loc": torch.tensor(
                emm_tips["X"], requires_grad=True, dtype=torch.float64
            )
        }
        self.ln_p = self.compute_ln_likelihood()
        self.current_epoch = 0
        assert prior in ("None", "birth_death", "gammadir"), "Invalid prior requested."
        self.prior = prior
        self.ln_prior = self.compute_ln_prior()
        self.matsumoto = matsumoto
        self.loss = torch.zeros(1)

    def learn(self, epochs, learn_rate, path_write):
        """Optimise params["dists"]"."""

        def lr_lambda(epoch):
            return 1.0 / (epoch + 1.0) ** 0.5

        optimizer = torch.optim.Adam(params=list(self.params.values()), lr=learn_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        post_hist = []

        if path_write is not None:
            emm_path = os.path.join(path_write, "embedding")
            post_path = os.path.join(path_write, "posterior.txt")
            os.mkdir(emm_path)

        def closure():
            optimizer.zero_grad()
            self.loss = -self.compute_ln_likelihood() - self.compute_ln_prior()
            self.loss.backward()
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
            if path_write is not None:
                self.save_epoch(i, emm_path, post_path)

        if path_write is not None:
            tree.save_tree_head(path_write, "mape", self.tip_labels)
            tree.save_tree(
                path_write,
                "mape",
                self.peel,
                self.blens,
                i + 1,
                self.ln_p.item(),
                self.ln_prior.item(),
            )

        if epochs > 0 and path_write is not None:
            HMAP.trace(epochs, post_hist, path_write)

    def save_epoch(self, i, emm_path, post_path):
        "Save posterior value and leaf locations to file."
        print_header = str([f"dim{i}, " for i in range(self.D)])
        ln_p = self.ln_p.item()
        ln_prior = self.ln_prior.item()
        ln_post = ln_p + ln_prior
        with open(post_path, "a", encoding="UTF-8") as file:
            file.write(f"{ln_post} = {ln_p} + {ln_prior}\n")
        emm_fn = os.path.join(emm_path, f"dists_hist_{i}.txt")
        np.savetxt(
            emm_fn,
            self.params["leaf_loc"].detach().numpy(),
            delimiter=", ",
            header=print_header,
        )

    def compute_ln_likelihood(self):
        """Compute likelihood of current tree, reducing soft_temp as required."""
        dist_2d = Chyp_torch.get_pdm(
            self.params["leaf_loc"], curvature=self.curvature, matsumoto=self.matsumoto
        )

        if self.loss_fn == "likelihood":
            self.peel, self.blens = peeler.nj_torch(dist_2d, tau=self.soft_temp)
            self.ln_p = self.compute_LL(self.peel, self.blens)
            loss = self.ln_p
        elif self.loss_fn == "pair_likelihood":
            self.peel, self.blens = peeler.nj_torch(dist_2d, tau=None)
            self.ln_p = self.compute_LL(self.peel, self.blens)
            loss = self.compute_log_a_like(dist_2d)
        return loss

    def compute_ln_prior(self):
        """Compute prior of current tree."""
        if self.prior == "None":
            return torch.zeros(1)
        elif self.prior == "gammadir":
            return self.compute_prior_gamma_dir(self.blens)
        elif self.prior == "birthdeath":
            return self.compute_prior_birthdeath(self.peel, self.blens)

    def get_ln_posterior(self, leaf_loc_flat):
        """Returns the posterior value.

        Assumes phylo likelihood as model.

        Args:
            leaf_loc ([type]): [description]
            curvature ([type]): [description]
        """
        leaf_loc = leaf_loc_flat.view((self.S, self.D))
        dist_2d = Chyp_torch.get_pdm(
            leaf_loc, curvature=self.curvature, matsumoto=self.matsumoto
        )
        peel, blens = peeler.nj_torch(dist_2d, tau=self.soft_temp)
        ln_p = self.compute_LL(peel, blens)
        ln_prior = self.compute_prior_gamma_dir(blens)
        return ln_p + ln_prior

    def laplace(self, path_write, n_samples=100):
        """Generate a laplace approximation around the current embedding.

        Args:
            path_write (string): Save trees in this directory.
            n (int, optional): Number of samples. Defaults to 100.
        """
        hessian = torch.autograd.functional.hessian
        normal = torch.distributions.multivariate_normal.MultivariateNormal

        print("Generating laplace approximation: ", end="", flush=True)
        filename = "laplace_samples"
        tree.save_tree_head(path_write, filename, self.tip_labels)
        mean = self.params["leaf_loc"].view(-1)
        for smp_i in range(n_samples):
            res = hessian(self.get_ln_posterior, mean, vectorize=True)
            cov = -torch.linalg.inv(res)
            norm_aprx = normal(mean, cov)
            sample = norm_aprx.sample(torch.Size((1,)))
            dists = Chyp_torch.get_pdm(
                sample, curvature=self.curvature, matsumoto=self.matsumoto
            )
            peel, blens = peeler.nj_torch(dists)
            blens = torch.tensor(blens)
            ln_p = self.compute_LL(peel, blens)
            ln_prior = self.compute_prior_gamma_dir(blens)
            if path_write is not None:
                tree.save_tree(
                    path_write,
                    filename,
                    peel,
                    blens,
                    smp_i,
                    ln_p,
                    ln_prior,
                    tip_labels=self.tip_labels,
                )
        print("done.")
