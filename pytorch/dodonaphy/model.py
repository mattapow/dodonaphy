from typing import List, Any

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .hyperboloid import t02p
from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc
import matplotlib.pyplot as plt


class DodonaphyModel(object):

    def __init__(self, partials, weights, dim):
        # self.parameters = {
        #     # radial distance
        #     "int_r": torch.empty(S-2, requires_grad=True),
        #     "int_dir": torch.empty(S-2, D, requires_grad=True),  # angles
        #     # adial distance of each tip sequence in the embedding
        #     "leaf_r": torch.empty(S, requires_grad=True),
        #     # directional coordinates of each tip sequence in the embedding
        #     "leaf_dir": torch.empty(S, D, requires_grad=True)
        # }
        self.partials = partials
        self.weights = weights
        self.S = len(partials)
        self.L = partials[0].shape[1]
        self.D = dim
        self.bcount = 2 * self.S - 2
        # Store mu on poincare ball in R^dim.
        # Distributions stored in tangent space T_0 H^D, then transformed to poincare ball.
        # The distribution for each point has a single sigma (i.e. mean field in x, y).
        leaf_x_sigma = np.abs(np.random.randn(self.S, self.D))
        int_x_sigma = np.abs(np.random.randn(self.S - 2))
        self.VariationalParams = {
            "leaf_x_mu": torch.randn((self.S, self.D), requires_grad=True, dtype=torch.float64),
            "leaf_x_sigma": torch.tensor(leaf_x_sigma, requires_grad=True, dtype=torch.float64),
            "int_x_mu": torch.randn((self.S - 2, self.D), requires_grad=True, dtype=torch.float64),
            "int_x_sigma": torch.tensor(int_x_sigma, requires_grad=True, dtype=torch.float64)
        }
        # make space for internal partials
        for i in range(self.S - 1):
            self.partials.append(torch.zeros((1, 4, self.L), dtype=torch.float64, requires_grad=False))


    def compute_branch_lengths(self, S, D, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
        """Computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball

        Args:
            S (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        blens = torch.empty(self.bcount, dtype=torch.float64)
        for b in range(S-1):
            directional2 = int_dir[peel[b][2]-S-1,]
            r2 = int_r[peel[b][2]-S-1]

            for i in range(2):
                if peel[b][i] < S:
                    # leaf to internal
                    r1 = leaf_r[peel[b][i]]
                    directional1 = leaf_dir[peel[b][i], :]
                else:
                    # internal to internal
                    r1 = int_r[peel[b][i]-S-1]
                    directional1 = int_dir[peel[b][i]-S-1,]

                hd = utilFunc.hyperbolic_distance(
                    r1, r2, directional1, directional2, curvature)

                # apply the inverse transform from Matsumoto et al 2020
                # add a tiny amount to avoid zero-length branches
                blens[peel[b][i]] = torch.log(torch.cosh(hd)) + 0.000000000001

        return blens

    def compute_LL(self, leaf_r, leaf_dir, int_r, int_dir):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """

        with torch.no_grad():
            peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)

        # branch lengths
        blens = self.compute_branch_lengths(
            self.S, self.D, peel, leaf_r, leaf_dir, int_r, int_dir)

        mats = JC69_p_t(blens)
        return calculate_treelikelihood(self.partials, self.weights, peel, mats,
                                        torch.full([4], 0.25, dtype=torch.float64))

    def draw_sample(self, nSample=100, **kwargs):
        """Draw samples from variational posterior distribution

        Args:
            nSample (int, optional): Number of samples to be drawn. Defaults to 100.

        Returns:
            [type]: [description]
        """
        # make peel, blens and X for each of these samples
        peel = []
        blens = []
        location = []
        lp__ = []
        for _ in range(nSample):
            # q_thetas in tangent space at origin in R^dim. Each point i, has a multivariate normal in dim=D
            q_leaf_x = []
            q_int_x = []
            for i in range(self.S):
                cov = self.VariationalParams["leaf_x_sigma"][i] * torch.eye(self.D)
                q_leaf_x.append(MultivariateNormal(torch.zeros(self.D).double(), cov.double()))
            for i in range(self.S - 2):
                cov = self.VariationalParams["int_x_sigma"][i] * torch.eye(self.D)
                q_int_x.append(MultivariateNormal(torch.zeros(self.D).double(), cov.double()))

            # Sample z in tangent space of hyperboloid at origin T_0 H^n
            z_leaf_x = torch.zeros((self.S, self.D))
            z_int_x = torch.zeros((self.S - 2, self.D))
            for i in range(self.S):
                z_leaf_x[i, :] = q_leaf_x[i].rsample((1,))
            for i in range(self.S - 2):
                z_int_x[i, :] = q_int_x[i].rsample((1,))

            # Tangent space at origin to Poincare
            z_leaf_x_poin = t02p(z_leaf_x, self.VariationalParams["leaf_x_mu"])
            z_int_x_poin = t02p(z_int_x, self.VariationalParams["int_x_mu"])

            # transform z to r, dir
            leaf_r, leaf_dir = utilFunc.cart_to_dir(z_leaf_x_poin)
            int_r, int_dir = utilFunc.cart_to_dir(z_int_x_poin)

            # prepare return (peel, branch lengths, locations, and log posteriori)
            pl = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
            peel.append(pl)
            bl = self.compute_branch_lengths(self.S, self.D, pl, leaf_r, leaf_dir, int_r, int_dir)
            blens.append(bl)
            location.append(utilFunc.dir_to_cart_tree(leaf_r, int_r, leaf_dir, int_dir, self.D))
            if kwargs.get('lp'):
                lp__.append(calculate_treelikelihood(self.partials, self.weights, pl, JC69_p_t(bl), torch.full([4], 0.25, dtype=torch.float64)))

        if kwargs.get('lp'):
            return peel, blens, location, lp__
        else:
            return peel, blens, location

    def calculate_elbo(self, q_leaf_x, q_int_x):
        """Calculate the elbo of a sample from the variational distributions q

        Args:
            q_leaf_x (Multivariate distribution):
                Distributions of leaves centred at origin in tangent space of hyperboloid T_0 H^n
            q_int_x (Multivariate distribution):
                Distributions of internal nodes centred at origin in tangent space of hyperboloid T_0 H^n

        Returns:
            float: The evidence lower bound of a sample from q
        """

        # Mean of distributions in R^dim+1
        # Convert Mean of distributions from Poincare to hyperboloid in R^dim+1

        # z in tangent space at origin
        z_leaf_x = torch.zeros((self.S, self.D))
        z_int_x = torch.zeros((self.S - 2, self.D))
        for i in range(self.S):
            z_leaf_x[i, :] = q_leaf_x[i].rsample((1,))
        for i in range(self.S - 2):
            z_int_x[i, :] = q_int_x[i].rsample((1,))

        # From (Euclidean) tangent space at origin to Poincare ball
        z_leaf_x_poin = t02p(z_leaf_x, self.VariationalParams["leaf_x_mu"])
        z_int_x_poin = t02p(z_int_x, self.VariationalParams["int_x_mu"])

        leaf_r, leaf_dir = utilFunc.cart_to_dir(z_leaf_x_poin)
        int_r, int_dir = utilFunc.cart_to_dir(z_int_x_poin)

        # Get Jacobians
        log_abs_det_jacobian = torch.zeros(1)
        # Leaves
        for i in range(self.S):
            # Jacobian of t02p going from Tangent T_0 to Poincare ball
            inputs = (z_leaf_x[i, :], self.VariationalParams["leaf_x_mu"][i, :])
            J_leaf = torch.autograd.functional.jacobian(t02p, inputs)
            for j in range(len(J_leaf)):
                log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(torch.det(J_leaf[j])))
        # Jacobian of going to polar
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(1/leaf_r).sum(0)

        # Internal nodes
        for i in range(self.S-2):
            inputs = (z_int_x[i, :], self.VariationalParams["int_x_mu"][i, :])
            J_int = torch.autograd.functional.jacobian(t02p, inputs)
            for j in range(len(J_int)):
                log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(torch.det(J_int[j])))
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(1 / int_r).sum(0)

        # logQ
        logQ = 0
        for i in range(self.S):
            logQ = logQ + q_leaf_x[i].log_prob(z_leaf_x[i])
        for i in range(self.S - 2):
            logQ = logQ + q_int_x[i].log_prob(z_int_x[i])

        # logPrior
        # TODO: have to think carefully
        logPrior = torch.zeros(1, requires_grad=False)

        logP = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)

        return logP + logPrior - logQ + log_abs_det_jacobian

    def learn(self, param_init=None, epochs=1000):
        """[summary]

        Args:
            param_init ([type]): [description]
            epochs (int, optional): [description]. Defaults to 1000.
        """
        if param_init is not None:
            # set initial params as a Dict
            self.VariationalParams["leaf_x_mu"] = param_init["leaf_x_mu"]
            self.VariationalParams["leaf_x_sigma"] = param_init["leaf_x_sigma"]
            self.VariationalParams["int_x_mu"] = param_init["int_x_mu"]
            self.VariationalParams["int_x_sigma"] = param_init["int_x_sigma"]

        lr_lambda = lambda epoch: 1.0 / np.sqrt(epoch + 1)
        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []
        for epoch in range(epochs):
            loss = - self.elbo_normal(3)
            elbo_hist.append(- loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            print('epoch {} ELBO: {}'.format(epoch, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

        plt.figure()
        plt.plot(range(epochs), elbo_hist, 'r', label='elbo')
        plt.title('Elbo values')
        plt.xlabel('Epochs')
        plt.ylabel('elbo')
        plt.legend()
        plt.show()

        # plt.hist(hist_dat)
        # plt.show()

        #with torch.no_grad(): # need grad for Jacobian in elbo
        print('Final ELBO: {}'.format(self.elbo_normal(100).item()))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        # q_thetas in tangent space at origin in T_0 H^dim. Each point i, has a multivariate normal in dim=D
        q_leaf_x = []
        q_int_x = []
        for i in range(self.S):
            cov = self.VariationalParams["leaf_x_sigma"][i] * torch.eye(self.D)
            q_leaf_x.append(MultivariateNormal(torch.zeros(self.D).double(), cov.double()))
        for i in range(self.S - 2):
            cov = self.VariationalParams["int_x_sigma"][i] * torch.eye(self.D)
            q_int_x.append(MultivariateNormal(torch.zeros(self.D).double(), cov.double()))

        elbos = []
        for _ in range(size):
            elbos.append(self.calculate_elbo(q_leaf_x, q_int_x))
        return torch.mean(torch.stack(elbos))
