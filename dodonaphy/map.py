"""Maximum A Posteriori of Distance matrix"""
import os

import numpy as np
import torch

from dodonaphy import peeler, tree
from dodonaphy.base_model import BaseModel


class MAP(BaseModel):
    """Maximum A Posteriori class for estimating the distance matrix."""

    def __init__(
        self,
        partials,
        weights,
        dists,
        soft_temp,
        loss_fn,
        prior="None",
        tip_labels=None,
    ):
        super().__init__(
            partials,
            weights,
            dim=None,
            soft_temp=soft_temp,
            connector="nj",
            curvature=-1,
            loss_fn=loss_fn,
            tip_labels=tip_labels,
        )
        tril_idx = torch.tril_indices(self.S, self.S, -1)
        dists_1d = dists[tril_idx[0], tril_idx[1]]
        self.params = {
            "dists": torch.tensor(dists_1d, requires_grad=True, dtype=torch.float64)
        }
        self.ln_p = self.compute_ln_likelihood()
        self.current_epoch = 0
        assert prior in ("None", "birth_death", "gammadir"), "Invalid prior requested."
        self.prior = prior
        self.ln_prior = self.compute_ln_prior()

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def learn(self, epochs, learn_rate, path_write):
        """Optimise params["dists"]"."""

        def lr_lambda(epoch):
            return 1.0 / (epoch + 1.0) ** 0.5

        optimizer = torch.optim.Adam(params=list(self.params.values()), lr=learn_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        post_hist = []

        if path_write is not None:
            post_path = os.path.join(path_write, "posterior.txt")
            dist_path = os.path.join(path_write, "dists")
            os.mkdir(dist_path)
            tree.save_tree_head(path_write, "samples", self.tip_labels)

        def closure():
            optimizer.zero_grad()
            loss = -self.compute_ln_likelihood() - self.compute_ln_prior()
            loss.backward()
            return loss

        print(f"Running for {epochs} iterations.")
        print("Iteration: log prior + log_likelihood = log posterior")
        for i in range(epochs):
            self.update_epoch(i)
            optimizer.step(closure)
            scheduler.step()
            post_hist.append(self.ln_p.item() + self.ln_prior.item())
            print(
                f"{i+1}: {self.ln_prior.item():.3f} + {self.ln_p.item():.3f} = {post_hist[-1]:.3f}"
            )
            if int(i + 1) % 10 == 9:
                print("")

            if path_write is not None:
                tree.save_tree(
                    path_write,
                    "samples",
                    self.peel,
                    self.blens,
                    i,
                    self.ln_p.item(),
                    self.ln_prior.item(),
                )
                with open(post_path, "a", encoding="UTF-8") as file:
                    file.write(f"{post_hist[-1]}\n")
                dists_fn = os.path.join(dist_path, f"dists_hist_{i}.txt")
                np.savetxt(
                    dists_fn,
                    optimizer.param_groups[0]["params"][0].detach().numpy(),
                    delimiter=", ",
                )

        if epochs > 0 and path_write is not None:
            MAP.trace(epochs, post_hist, path_write)

    def compute_ln_likelihood(self):
        """Compute likelihood of current tree, reducing soft_temp as required."""
        tril_idx = torch.tril_indices(self.S, self.S, -1)
        dist_2d = torch.zeros((self.S, self.S), dtype=torch.double)
        dist_2d[tril_idx[0], tril_idx[1]] = self.params["dists"]
        dist_2d[tril_idx[1], tril_idx[0]] = self.params["dists"]

        if self.loss_fn == "likelihood":
            self.peel, self.blens = peeler.nj_torch(dist_2d, tau=self.soft_temp)
            self.ln_p = self.compute_LL(self.peel, self.blens)
            loss = self.ln_p
        elif self.loss_fn == "pair_likelihood":
            self.peel, self.blens = peeler.nj_torch(dist_2d, tau=None)
            self.ln_p = self.compute_LL(self.peel, self.blens)
            loss = self.compute_log_a_like(dist_2d)
        elif self.loss_fn == "hypHC":
            raise ValueError("hypHC requires embedding, not available with MAP.")
        return loss

    def compute_ln_prior(self):
        """Compute prior of current tree."""
        if self.prior == "None":
            return torch.zeros(1)
        elif self.prior == "gammadir":
            return self.compute_prior_gamma_dir(self.blens)
        elif self.prior == "birthdeath":
            return self.compute_prior_birthdeath(self.peel, self.blens)
