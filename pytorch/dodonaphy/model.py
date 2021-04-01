import numpy as np
import torch
import statistics
import matplotlib as plt
from torch.distributions import SigmoidTransform

from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc
import matplotlib.pyplot as plt


class DodonaphyModel(object):

    def __init__(self, partials, weights, D):
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
        self.D = D
        self.bcount = 2 * self.S - 2
        self.VarationalParams = {
            "leaf_r_mu": torch.randn(self.S, requires_grad=True, dtype=torch.float64),
            "leaf_r_sigma": torch.randn(self.S, requires_grad=True, dtype=torch.float64),
            "leaf_dir_mu": torch.randn(self.S, D, requires_grad=True, dtype=torch.float64),
            "leaf_dir_sigma": torch.randn(self.S, D, requires_grad=True, dtype=torch.float64),
            "int_r_mu": torch.randn(self.S - 2, requires_grad=True, dtype=torch.float64),
            "int_r_sigma": torch.randn(self.S - 2, requires_grad=True, dtype=torch.float64),
            "int_dir_mu": torch.randn(self.S - 2, D, requires_grad=True, dtype=torch.float64),
            "int_dir_sigma": torch.randn(self.S - 2, D, requires_grad=True, dtype=torch.float64)
        }
        # self.VarationalParams = {
        #     "leaf_r_mu": torch.tensor(np.array(self.S*[0]), requires_grad=True, dtype=torch.float64),
        #     "leaf_r_sigma": torch.tensor(np.array(self.S*[1]), requires_grad=True, dtype=torch.float64),
        #     "leaf_dir_mu": torch.tensor(np.array(self.S*[D*[0]]), requires_grad=True, dtype=torch.float64),
        #     "leaf_dir_sigma": torch.tensor(np.array(self.S*[D*[1]]), requires_grad=True, dtype=torch.float64),
        #     "int_r_mu": torch.tensor(np.array((self.S-2)*[0]), requires_grad=True, dtype=torch.float64),
        #     "int_r_sigma": torch.tensor(np.array((self.S-2)*[1]), requires_grad=True, dtype=torch.float64),
        #     "int_dir_mu": torch.tensor(np.array((self.S-2)*[D*[0]]), requires_grad=True, dtype=torch.float64),
        #     "int_dir_sigma": torch.tensor(np.array((self.S-2)*[D*[1]]), requires_grad=True, dtype=torch.float64)
        # }
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
        for b in range(S-1):
            directional1, directional2 = torch.empty(
                D, requires_grad=False), torch.empty(D, requires_grad=False)
            directional2 = int_dir[peel[b][2]-S-1,]
            r1 = torch.empty(1)
            r2 = int_r[peel[b][2]-S-1]
            if peel[b][0] < S:
                # leaf to internal
                r1 = leaf_r[peel[b][0]]
                directional1 = leaf_dir[peel[b][0], :]
            else:
                # internal to internal
                r1 = int_r[peel[b][0]-S-1]
                directional1 = int_dir[peel[b][0]-S-1,]
            # apply the inverse transform from Matsumoto et al 2020
            # add a tiny amount to avoid zero-length branches
            blens[peel[b][0]] = torch.log(
                torch.cosh(blens[peel[b][0]])) + 0.000000000001

            if peel[b][1] < S:
                # leaf to internal
                r1 = leaf_r[peel[b][1]]
                directional1 = leaf_dir[peel[b][1], ]
            else:
                # internal to internal
                r1 = int_r[peel[b][1]-S-1]
                directional1 = int_dir[peel[b][1]-S-1,]
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

        # brach lenghts
        blens = self.compute_branch_lengths(
            self.S, self.D, peel, leaf_r, leaf_dir, int_r, int_dir)

        mats = JC69_p_t(blens)
        return calculate_treelikelihood(self.partials, self.weights, peel, mats, torch.full([4], 0.25, dtype=torch.float64))

    def draw_sample(self, nSample=100):
        """[summary]

        Args:
            nSample (int, optional): [description]. Defaults to 100.

        Returns:
            [type]: [description]
        """
        sigmoid_transformation = SigmoidTransform()

        # q_thetas
        q_leaf_r = torch.distributions.Normal(
            self.VarationalParams["leaf_r_mu"],
            self.VarationalParams["leaf_r_sigma"].exp())
        q_leaf_dir = torch.distributions.Normal(
            self.VarationalParams["leaf_dir_mu"],
            self.VarationalParams["leaf_dir_sigma"].exp())
        q_int_r = torch.distributions.Normal(
            self.VarationalParams["int_r_mu"],
            self.VarationalParams["int_r_sigma"].exp())
        q_int_dir = torch.distributions.Normal(
            self.VarationalParams["int_dir_mu"],
            self.VarationalParams["int_dir_sigma"].exp())
        # z
        z_leaf_r = q_leaf_r.rsample((nSample,))
        z_leaf_dir = q_leaf_dir.rsample((nSample,))
        z_int_r = q_int_r.rsample((nSample,))
        z_int_dir = q_int_dir.rsample((nSample,))

        # transformation of z
        leaf_r = sigmoid_transformation(z_leaf_r)
        leaf_dir = sigmoid_transformation(z_leaf_dir) * 2 - 1
        int_r = sigmoid_transformation(z_int_r)
        int_dir = sigmoid_transformation(z_int_dir) * 2 - 1

        # take transformations into account
        log_abs_det_jacobian = sigmoid_transformation.log_abs_det_jacobian(leaf_r, z_leaf_r).sum() + \
                               sigmoid_transformation.log_abs_det_jacobian(int_r, z_int_r).sum() + \
                               sigmoid_transformation.log_abs_det_jacobian(leaf_dir, z_leaf_dir).sum() * 2.0 + \
                               sigmoid_transformation.log_abs_det_jacobian(int_dir, z_int_dir).sum() * 2.0

        # logQ
        logQ_leaf_r = q_leaf_r.log_prob(z_leaf_r)
        logQ_leaf_dir = q_leaf_dir.log_prob(z_leaf_dir)
        logQ_int_r = q_int_r.log_prob(z_int_r)
        logQ_int_dir = q_int_dir.log_prob(z_int_dir)

        
        # make peel and blens, and posterior likelihood for each of these samples
        peel = []
        blens = []
        lp__ = []
        for i in range(nSample):
            pl = utilFunc.make_peel(logQ_leaf_r[i], logQ_leaf_dir[i], logQ_int_r[i], logQ_int_dir[i])
            bl = self.compute_branch_lengths(self.S, self.D, pl,logQ_leaf_r[i], logQ_leaf_dir[i], logQ_int_r[i], logQ_int_dir[i])
            mats = JC69_p_t(bl)
            lp = calculate_treelikelihood(self.partials, self.weights, pl, mats, torch.full([4], 0.25, dtype=torch.float64))
            peel.append(pl)
            blens.append(bl)
            lp__.append(lp)

        return peel, blens, lp__

    def calculate_elbo(self, q_leaf_r, q_leaf_dir, q_int_r, q_int_dir):
        """[summary]

        Args:
            q_leaf_r ([type]): [description]
            q_leaf_dir ([type]): [description]
            q_int_r ([type]): [description]
            q_int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        sigmoid_transformation = SigmoidTransform()

        # z
        z_leaf_r = q_leaf_r.rsample()
        z_leaf_dir = q_leaf_dir.rsample()
        z_int_r = q_int_r.rsample()
        z_int_dir = q_int_dir.rsample()

        # transformation of z
        leaf_r = sigmoid_transformation(z_leaf_r)
        leaf_dir = sigmoid_transformation(z_leaf_dir) * 2 - 1
        int_r = sigmoid_transformation(z_int_r)
        int_dir = sigmoid_transformation(z_int_dir) * 2 - 1

        # take transformations into account
        log_abs_det_jacobian = sigmoid_transformation.log_abs_det_jacobian(leaf_r, z_leaf_r).sum() + \
                               sigmoid_transformation.log_abs_det_jacobian(int_r, z_int_r).sum() + \
                               sigmoid_transformation.log_abs_det_jacobian(leaf_dir, z_leaf_dir).sum() * 2.0 + \
                               sigmoid_transformation.log_abs_det_jacobian(int_dir, z_int_dir).sum() * 2.0

        # logQ
        logQ_leaf_r = q_leaf_r.log_prob(z_leaf_r)
        logQ_leaf_dir = q_leaf_dir.log_prob(z_leaf_dir)
        logQ_int_r = q_int_r.log_prob(z_int_r)
        logQ_int_dir = q_int_dir.log_prob(z_int_dir)
        logQ = logQ_leaf_r.sum() + logQ_leaf_dir.sum() + logQ_int_r.sum() + logQ_int_dir.sum()

        # logPrior, have to think carefully
        logPrior = torch.zeros(1, requires_grad=True)

        logP = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)

        return logP + logPrior  - logQ + log_abs_det_jacobian

    def learn(self, param_init=None, epochs=1000):
        """[summary]

        Args:
            dpy_dat ([type]): [description]
            param_init ([type]): [description]
            epoch (int, optional): [description]. Defaults to 1000.
        """
        if param_init is not None:
            # set initial params as a Dict
            self.VarationalParams["leaf_r_mu"], self.VarationalParams["leaf_r_sigma"] = torch.tensor(
                param_init["leaf_r"].mean(), requires_grad=True), torch.tensor(param_init["leaf_r"].std(), requires_grad=True)

            self.VarationalParams["leaf_dir_mu"], self.VarationalParams["leaf_dir_sigma"] = torch.tensor(
                param_init["leaf_dir"].mean(
                ), requires_grad=True), torch.tensor(param_init["leaf_dir"].std(), requires_grad=True)
            self.VarationalParams["int_r_mu"], self.VarationalParams["int_r_sigma"] = torch.tensor(
                param_init["int_r"].mean(
                ), requires_grad=True), torch.tensor(param_init["int_r"].std(), requires_grad=True)
            self.VarationalParams["int_dir_mu"], self.VarationalParams["int_dir_sigma"] = torch.tensor(
                param_init["int_dir"].mean(
                ), requires_grad=True), torch.tensor(param_init["int_dir"].std(), requires_grad=True)

        lr_lambda = lambda epoch: 1.0 / np.sqrt(epoch + 1)
        optimizer = torch.optim.Adam(list(self.VarationalParams.values()), lr=0.1)
        # optimizer = torch.optim.Adam((self.VarationalParams["leaf_r_mu"],
        # self.VarationalParams["leaf_r_sigma"],
        # self.VarationalParams["leaf_dir_mu"],
        # self.VarationalParams["leaf_dir_sigma"],
        # self.VarationalParams["int_r_mu"],
        # self.VarationalParams["int_r_sigma"],
        # self.VarationalParams["int_dir_mu"],
        # self.VarationalParams["int_dir_sigma"]), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        
        hist_dat = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            loss = - self.elbo_normal()
            elbo_hist.append(- loss.item())
            loss.backward()

            optimizer.step()
            scheduler.step()
            
            print('epoch {} ELBO: {}'.format(epoch, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

        plt.plot(range(epochs), elbo_hist, 'r', label='elbo')
        plt.title('Elbo values')
        plt.xlabel('Epochs')
        plt.ylabel('elbo')
        plt.legend()
        plt.show()
            

        # plt.hist(hist_dat)

        with torch.no_grad():
            print('Final ELBO: {}'.format(self.elbo_normal(100).item()))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        # q_thetas
        q_leaf_r = torch.distributions.Normal(
            self.VarationalParams["leaf_r_mu"],
            self.VarationalParams["leaf_r_sigma"].exp())
        q_leaf_dir = torch.distributions.Normal(
            self.VarationalParams["leaf_dir_mu"],
            self.VarationalParams["leaf_dir_sigma"].exp())
        q_int_r = torch.distributions.Normal(
            self.VarationalParams["int_r_mu"],
            self.VarationalParams["int_r_sigma"].exp())
        q_int_dir = torch.distributions.Normal(
            self.VarationalParams["int_dir_mu"],
            self.VarationalParams["int_dir_sigma"].exp())

        # elbos = []
        # for i in range(size):
        #     elbos.append(self.calculate_elbo(q_leaf_r, q_leaf_dir, q_int_r, q_int_dir))
        # return torch.mean(torch.tensor(elbos, requires_grad=True))
        return self.calculate_elbo(q_leaf_r, q_leaf_dir, q_int_r, q_int_dir)
