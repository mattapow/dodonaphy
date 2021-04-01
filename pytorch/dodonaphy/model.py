from typing import List, Any

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .hyperboloid import transform_to_hyper, hyper_to_poincare, poincare_to_hyper
from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc


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
        self.VariationalParams = {
            "leaf_x_mu": torch.zeros((self.S, self.D), requires_grad=True, dtype=torch.float64),
            "leaf_x_sigma": torch.ones(self.S, requires_grad=True, dtype=torch.float64),
            "int_x_mu": torch.zeros((self.S - 2, self.D), requires_grad=True, dtype=torch.float64),
            "int_x_sigma": torch.ones((self.S - 2), requires_grad=True, dtype=torch.float64)
        }

        # make space for internal partials
        for i in range(self.S - 1):
            self.partials.append([None] * self.L)

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
        for b in range(S - 1):
            directional1, directional2 = torch.empty(
                D, requires_grad=False), torch.empty(D, requires_grad=False)
            directional2 = int_dir[peel[b][2] - S - 1, ]
            r1 = torch.empty(1)
            r2 = int_r[peel[b][2] - S - 1]
            if peel[b][0] < S:
                # leaf to internal
                r1 = leaf_r[peel[b][0]]
                directional1 = leaf_dir[peel[b][0], :]
            else:
                # internal to internal
                r1 = int_r[peel[b][0] - S - 1]
                directional1 = int_dir[peel[b][0] - S - 1,]
            # apply the inverse transform from Matsumoto et al 2020
            # add a tiny amount to avoid zero-length branches
            blens[peel[b][0]] = torch.log(
                torch.cosh(blens[peel[b][0]])) + 0.000000000001

            if peel[b][1] < S:
                # leaf to internal
                r1 = leaf_r[peel[b][1]]
                directional1 = leaf_dir[peel[b][1],]
            else:
                # internal to internal
                r1 = int_r[peel[b][1] - S - 1]
                directional1 = int_dir[peel[b][1] - S - 1,]
            blens[peel[b][1]] = utilFunc.hyperbolic_distance(
                r1, r2, directional1, directional2, curvature)

            # apply the inverse transform from Matsumoto et al 2020
            # add a tiny amount to avoid zero-length branches
            blens[peel[b][1]] = torch.log(
                torch.cosh(blens[peel[b][1]])) + 0.000000000001

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

    def draw_sample(self, nSample=100):
        """[summary]

        Args:
            nSample (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]
        """
        # make peel, blens and X for each of these samples
        peel = []
        blens = []
        location = []
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

            # Convert Mean of distributions from Poincare to hyperboloid in R^dim+1
            leaf_loc_hyp = poincare_to_hyper(self.VariationalParams["leaf_x_mu"])
            int_loc_hyp = poincare_to_hyper(self.VariationalParams["int_x_mu"])

            # z in tangent space of hyperboloid at origin T_0 H^n
            z_leaf_x = torch.zeros((self.S, self.D))
            z_int_x = torch.zeros((self.S - 2, self.D))
            for i in range(self.S):
                z_leaf_x[i, :] = q_leaf_x[i].rsample((1,))
            for i in range(self.S - 2):
                z_int_x[i, :] = q_int_x[i].rsample((1,))

            # transform z from tangent space at origin to hyperboloid at mu
            z_leaf_x_hyp = torch.zeros((self.S, self.D + 1))
            z_int_x_hyp = torch.zeros((self.S - 2, self.D + 1))
            for i in range(self.S):
                z_leaf_x_hyp[i, :] = transform_to_hyper(leaf_loc_hyp[i, :], z_leaf_x[i, :], self.D)
            for i in range(self.S - 2):
                z_int_x_hyp[i, :] = transform_to_hyper(int_loc_hyp[i, :], z_int_x[i, :], self.D)

            # transform z from hyperboloid to poincare ball
            z_leaf_x_poin = torch.zeros((self.S, self.D))
            z_int_x_poin = torch.zeros((self.S - 2, self.D))
            for i in range(self.S):
                z_leaf_x_poin[i, :] = hyper_to_poincare(z_leaf_x_hyp[i, :])
            for i in range(self.S - 2):
                z_int_x_poin[i, :] = hyper_to_poincare(z_int_x_hyp[i, :])

            # transform z to r, dir
            leaf_r, leaf_dir = utilFunc.cart_to_dir(z_leaf_x_poin)
            int_r, int_dir = utilFunc.cart_to_dir(z_int_x_poin)

            pl = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
            peel.append(pl)
            blens.append(self.compute_branch_lengths(self.S, self.D, pl, leaf_r, leaf_dir, int_r, int_dir))
            location.append(utilFunc.dir_to_cart_tree(leaf_r, int_r, leaf_dir, int_dir, self.D))

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
        # TODO: vectorise for optimisation?
        # Mean of distributions in R^dim+1
        # Convert Mean of distributions from Poincare to hyperboloid in R^dim+1
        leaf_loc_hyp = poincare_to_hyper(self.VariationalParams["leaf_x_mu"])
        int_loc_hyp = poincare_to_hyper(self.VariationalParams["int_x_mu"])

        # z in tangent space at origin
        z_leaf_x = torch.zeros((self.S, self.D))
        z_int_x = torch.zeros((self.S - 2, self.D))
        for i in range(self.S):
            z_leaf_x[i, :] = q_leaf_x[i].rsample((1,))
        for i in range(self.S - 2):
            z_int_x[i, :] = q_int_x[i].rsample((1,))

        # transform z from tangent space at origin to hyperboloid at mu
        z_leaf_x_hyp = torch.zeros((self.S, self.D + 1))
        z_int_x_hyp = torch.zeros((self.S - 2, self.D + 1))
        for i in range(self.S):
            z_leaf_x_hyp[i, :] = transform_to_hyper(leaf_loc_hyp[i, :], z_leaf_x[i, :], self.D)
        for i in range(self.S - 2):
            z_int_x_hyp[i, :] = transform_to_hyper(int_loc_hyp[i, :], z_int_x[i, :], self.D)

        # transform z to poincare ball
        z_leaf_x_poin = torch.zeros((self.S, self.D))
        z_int_x_poin = torch.zeros((self.S - 2, self.D))
        for i in range(self.S):
            z_leaf_x_poin[i, :] = hyper_to_poincare(z_leaf_x_hyp[i, :])
        for i in range(self.S - 2):
            z_int_x_poin[i, :] = hyper_to_poincare(z_int_x_hyp[i, :])

        # transform z to r, dir
        leaf_r, leaf_dir = utilFunc.cart_to_dir(z_leaf_x_poin)
        int_r, int_dir = utilFunc.cart_to_dir(z_int_x_poin)

        # TODO: Compute Jacobian
        log_abs_det_jacobian = 1

        # logQ
        logQ = 0
        for i in range(self.S):
            logQ += q_leaf_x[i].log_prob(z_leaf_x[i])
        for i in range(self.S - 2):
            logQ += q_int_x[i].log_prob(z_int_x[i])

        # logPrior, have to think carefully
        logPrior = 0

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
        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []
        for epoch in range(epochs):
            loss = - self.elbo_normal()

            elbo_hist.append(-loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('epoch {} ELBO: {}'.format(epoch, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

        # plt.hist(hist_dat)
        # plt.show()

        with torch.no_grad():
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
        for i in range(size):
            elbos.append(self.calculate_elbo(q_leaf_x, q_int_x))
        return torch.mean(torch.tensor(elbos, requires_grad=True))
