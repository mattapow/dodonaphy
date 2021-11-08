from dodonaphy.base_model import BaseModel
from dodonaphy import peeler, tree
import torch
import os


class ML(BaseModel):
    """Maximum Likelihood class"""

    def __init__(self, partials, weights, dists=None, temp=None):
        self.temp = temp
        super().__init__(partials, weights, None, curvature=-1, dists=dists)

    def learn(self, epochs, lr, path_write):
        def lr_lambda(epoch):
            return 1.0 / (epoch + 1) ** 0.5

        optimizer = torch.optim.LBFGS(params=list(self.dists.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        like_hist = []

        def closure():
            optimizer.zero_grad()
            loss = -self.compute_likelihood()
            loss.backward()
            return loss

        for i in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            like_hist.append(self.lnP.item())
            print("epoch %i Likelihood: %.3f" % (i + 1, like_hist[-1]))

            if path_write is not None:
                tree.save_tree(path_write, "ml", self.peel, self.blens, i, self.lnP, -1)
                like_fn = os.path.join(path_write, "list_hist.txt")
                with open(like_fn, "a") as f:
                    f.write("%f\n" % like_hist[-1])

        if epochs > 0 and path_write is not None:
            ML.trace(epochs, like_hist, path_write)
        return

    def run(taxa, partials, weights, dists, path_write, epochs, lr, temp):
        dists_torch = torch.tensor(dists, requires_grad=True, dtype=torch.float64)
        mymod = ML(partials, weights, dists=dists_torch, temp=temp)
        mymod.learn(epochs=epochs, lr=lr, path_write=path_write)
        return

    def compute_likelihood(self):
        self.peel, self.blens = peeler.nj(self.dists["dists"], tau=self.temp)
        self.lnP = self.compute_LL(self.peel, self.blens)
        return self.lnP

    def trace(epochs, like_hist, path_write):
        try:
            import matplotlib.pyplot as plt

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
        except Exception:
            print("Could not generate and save likelihood figures.")
