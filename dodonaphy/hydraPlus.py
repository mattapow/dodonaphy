import numpy as np
from scipy.optimize import minimize

from dodonaphy import hydra, Cutils, Chyperboloid


class HydraPlus:
    dists = None
    dim = 0
    curvature = -1.0
    n_taxa = 0
    eps = np.finfo(np.double).eps

    def __init__(self, dists, dim, curvature=-1.0):
        self.dists = dists
        self.dim = dim
        self.curvature = curvature
        self.n_taxa = len(dists)

    def embed(self, alpha=1.1, equi_adj=0.5, maxiter=1000, **kwargs):
        # Embed the distance matrix into the Poincare ball using Hydra+

        print("Minimising initial embedding strain: ", end="")
        emm = hydra.hydra(
            self.dists,
            self.dim,
            curvature=self.curvature,
            alpha=alpha,
            equi_adj=equi_adj,
            **kwargs
        )
        loc_init = np.tile(emm["r"], (self.dim, 1)).T * emm["directional"]
        loc_init = Chyperboloid.poincare_to_hyper(loc_init).flatten()
        self.dim += 1
        print("done.")

        print("Minimising initial embedding stress: ", end="", flush=True)
        optimizer = minimize(
            self.get_stress,
            loc_init,
            method="BFGS",
            jac=self.get_stress_gradient,
            options={"disp": False, "maxiter": maxiter},
        )
        X = optimizer.x.reshape((self.n_taxa, self.dim))
        X = Chyperboloid.hyper_to_poincare(X.T).T
        self.dim -= 1
        print("done.", flush=True)

        output = {}
        output["X"] = X
        radius, directional = Cutils.cart_to_dir_np(X)
        output["r"] = radius
        output["directional"] = directional
        output["stress_hydra"] = emm["stress"]
        output["stress_hydraPlus"] = optimizer.fun
        output["curvature"] = self.curvature
        output["dim"] = self.dim
        return output

    def get_stress_gradient(self, x):
        # This function calculates the gradient for stress-minimzation
        # x is the vectorization of the coordinate matrix X, which has dimensions nrows x ncols.
        # The rows of X are the embedded points and the columns the reduced hyperbolic coordinates
        x = x.reshape((self.n_taxa, self.dim))
        X = np.matmul(x, x.T)
        u_tilde = np.sqrt(X.diagonal() + 1)
        H = X - np.outer(u_tilde, u_tilde)
        H = np.minimum(H, -(1 + self.eps))  # bad
        D = 1 / np.sqrt(-self.curvature) * np.arccosh(-H)  # bad
        np.fill_diagonal(D, 0)
        A = (D - self.dists) * (1 / np.sqrt(-self.curvature * (H ** 2 - 1)))
        np.fill_diagonal(A, 0)
        B = np.outer((1 / u_tilde), u_tilde)
        AB_sum = np.tile(np.sum(A * B, axis=1), (self.dim, 1)).T
        G = 2 * (AB_sum * x - A @ x)
        return G.flatten()

    def get_stress(self, x):
        x = x.reshape((self.n_taxa, self.dim))
        X = np.matmul(x, x.T)
        u_tilde = np.sqrt(X.diagonal() + 1)
        H = X - np.outer(u_tilde, u_tilde)
        D = 1 / np.sqrt(-self.curvature) * np.arccosh(np.maximum(-H, 1))
        np.fill_diagonal(D, 0)
        y = 0.5 * np.sum((D - self.dists) ** 2)
        return y

