import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dodonaphy import peeler, tree
from dodonaphy.base_model import BaseModel


class ML(BaseModel):
    """Maximum Likelihood class"""

    def __init__(self, partials, weights, dists, soft_temp):
        self.params = {"dists": dists}
        super().__init__(
            partials,
            weights,
            dim=None,
            soft_temp=soft_temp,
            connector="nj",
            curvature=-1,
        )
        self.ln_p = self.compute_likelihood()

    def learn(self, epochs, lr, path_write):
        def lr_lambda(epoch):
            return 1.0 / (epoch + 1) ** 0.5

        optimizer = torch.optim.LBFGS(params=list(self.params.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        like_hist = []

        if path_write is not None:
            like_fn = os.path.join(path_write, "list_hist.txt")
            dist_path = os.path.join(path_write, "dists")
            os.mkdir(dist_path)

        def closure():
            optimizer.zero_grad()
            loss = -self.compute_likelihood()
            loss.backward()
            return loss

        print(f"Running for {epochs} iterations.")
        for i in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            like_hist.append(self.ln_p.item())
            print(f"epoch {i+1} Likelihood: {like_hist[-1]:.3f}")

            if path_write is not None:
                tree.save_tree(
                    path_write, "ml", self.peel, self.blens, i, self.ln_p, -1
                )
                with open(like_fn, "a", encoding='UTF-8') as f:
                    f.write("%f\n" % like_hist[-1])
                dists_fn = os.path.join(dist_path, f"dists_hist_{i}.txt")
                np.savetxt(
                    dists_fn,
                    optimizer.param_groups[0]["params"][0].detach().numpy(),
                    delimiter=", ",
                )

        if epochs > 0 and path_write is not None:
            ML.trace(epochs, like_hist, path_write)
        return

    @staticmethod
    def run(taxa, partials, weights, dists, path_write, epochs, lr, soft_temp):
        tril_idx = torch.tril_indices(taxa, taxa, -1)
        dists_1d = dists[tril_idx[0], tril_idx[1]]
        dists_torch = torch.tensor(dists_1d, requires_grad=True, dtype=torch.float64)
        mymod = ML(partials, weights, dists=dists_torch, soft_temp=soft_temp)
        mymod.learn(epochs=epochs, lr=lr, path_write=path_write)
        return

    def compute_likelihood(self):
        tril_idx = torch.tril_indices(self.S, self.S, -1)
        dist_2d = torch.zeros((self.S, self.S), dtype=torch.double)
        dist_2d[tril_idx[0], tril_idx[1]] = self.params["dists"]
        dist_2d[tril_idx[1], tril_idx[0]] = self.params["dists"]

        good_peel=False
        while ~good_peel:
            self.peel, self.blens = peeler.nj(dist_2d, tau=self.soft_temp)
            set1 = set(np.sort(np.unique(self.peel)))
            set2 = set(np.arange(self.bcount + 1))
            if set1 != set2:
                print("Decreasing temperature by half.")
                self.soft_temp/=2
            else:
                good_peel=True
        
        self.ln_p = self.compute_LL(self.peel, self.blens)
        return self.ln_p

    @staticmethod
    def trace(epochs, like_hist, path_write):
        plt.figure()
        plt.plot(range(epochs), like_hist, "r", label="likelihood")
        plt.xlabel("Epochs")
        plt.ylabel("likelihood")
        plt.legend()
        plt.savefig(path_write + "/likelihood_trace.png")

        plt.clf()
        plt.hist(like_hist)
        plt.title("Likelihood histogram")
        plt.savefig(path_write + "/likelihood_hist.png")
