from dodonaphy.base_model import BaseModel
from dodonaphy import peeler, tree
import torch
import os


class ML(BaseModel):
    """Maximum Likelihood class
    """
    def __init__(self, partials, weights, dists=None, temp=None):
        self.temp=temp
        super().__init__(partials, weights, None, curvature=-1, dists=dists)

    def learn(self, epochs, lr, path_write):
        optimizer = torch.optim.LBFGS(params=list(self.dists.values()), lr=lr)
        like_hist = []

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = -self.compute_likelihood()
            loss.backward()
            like_hist.append(-loss.item())
            print("epoch %i Likelihood: %.3f" % (i + 1, -loss.item()))
            return loss

        for i in range(epochs):
            optimizer.step(closure)

            if path_write is not None:
                peel, blens = peeler.nj(self.dists)
                tree.save_tree(path_write, "ml", peel, blens, i, self.lnP, -1)
                like_fn = os.path.join(path_write, "list_hist.txt")
                with open(like_fn, "a") as f:
                    f.write("%f\n" % like_hist[-1])
        return

    def run(taxa, partials, weights, dists, path_write, epochs, lr):
        dists_torch = torch.tensor(dists, requires_grad=True, dtype=torch.float64)
        mymod = ML(partials, weights, dists=dists_torch)
        mymod.learn(epochs=epochs, lr=lr, path_write=path_write)
        return

    def compute_likelihood(self):
        peel, blens = peeler.nj(self.dists["dists"], tau=0.0001)
        self.lnP = self.compute_LL(peel, blens)
        return self.lnP
