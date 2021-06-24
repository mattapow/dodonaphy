import os
from typing import List, Any

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.transforms import SigmoidTransform

from .hyperboloid import t02p
from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc
from .base_model import BaseModel
from src.hyperboloid import p2t0
import matplotlib.pyplot as plt


class DodonaphyVI(BaseModel):

    def __init__(self, partials, weights, dim, boosts=1, method='wrap', connect_method='mst', **prior):
        super().__init__(partials, weights, dim, connect_method=connect_method, **prior)
        print('Initialising variational model.\n')

        # Store mu on poincare ball in R^dim.
        # Distributions stored in tangent space T_0 H^D, then transformed to poincare ball.
        # The distribution for each point has a single sigma (i.e. mean field in x, y).
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.random.randn(self.S, self.D)) + eps)
        int_sigma = np.log(np.abs(np.random.randn(self.S - 2, self.D)) + eps)
        self.boosts = boosts
        self.method = method
        self.connect_method = connect_method

        leaf_sigma = self.init_boosters(leaf_sigma, boosts)
        int_sigma = self.init_boosters(int_sigma, boosts)

        if self.connect_method == 'geodesics':
            self.VariationalParams = {
                "leaf_mu": torch.randn((boosts, self.S, self.D), requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
                "leaf_weights": torch.full((boosts, 1), 1. / boosts).requires_grad_(True)
            }
        elif self.connect_method == 'mst':
            self.VariationalParams = {
                "leaf_mu": torch.randn((boosts, self.S, self.D), requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
                "leaf_weights": torch.full((boosts, 1), 1. / boosts).requires_grad_(True),
                "int_mu": torch.randn((boosts, self.S-2, self.D),
                                                           requires_grad=True, dtype=torch.float64),
                "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64),
                "int_weights": torch.full((boosts, 1), 1. / boosts).requires_grad_(True)
            }
            
            

    def sample_loc(self, q):
        # Sample distribution q in tangent space of hyperboloid at origin T_0 H^n
        z = [q[k].rsample((1,)).squeeze() for k in range(self.boosts)]

        logQ = torch.zeros(1)
        for k in range(self.boosts):
            logQ = logQ + self.VariationalParams['leaf_weights'][k] * q[k].log_prob(z[k])

        z = sum(z)

        return z, logQ

    @staticmethod
    def init_boosters(x, boosts):
        # Repeat starting distributions for boosting
        shape = list(x.shape)
        shape.insert(0, boosts)
        x_boosted = np.zeros(shape)
        for k in range(boosts):
            x_boosted[k] = x / boosts
        return x_boosted

    def draw_sample(self, nSample=100, **kwargs):
        """Draw samples from the variational posterior distribution

        Args:
            nSample (int, optional): Number of samples to be drawn. Defaults to 100.

        Returns:
            tuple[list list list list]: peel, blens, location, lp. If kwarg 'lp' is passed.
            Locations are in Poincare disk. lp = log-probability
            tuple[list list list]: peel, blens, location, lp. Otherwise.
        """
        with torch.no_grad():
            # q_thetas in tangent space at origin in T_0 H^dim. Each point i, has a multivariate normal in dim=D
            q_leaves = []
            q_ints = []
            for k in range(self.boosts):
                mu = self.VariationalParams["leaf_mu"][k].reshape(self.S * self.D)
                cov = self.VariationalParams["leaf_sigma"][k].exp().reshape(self.S * self.D)*torch.eye(self.S * self.D)
                q_leaves.append(MultivariateNormal(mu, cov))

                if self.connect_method == 'mst':
                    mu = self.VariationalParams["int_mu"][k].reshape((self.S - 2) * self.D)
                    cov = self.VariationalParams["int_sigma"][k].exp().reshape((self.S - 2) * self.D)\
                        * torch.eye((self.S - 2) * self.D)
                    q_ints.append(MultivariateNormal(mu, cov))

            # make peel, blens and X for each of these samples
            peel = []
            blens = []
            location = []
            lp = []
            for _ in range(nSample):
                # Sample in tangent space
                z_leaf, _ = self.sample_loc(q_leaves)
                if self.connect_method == 'mst':
                    z_int, _ = self.sample_loc(q_ints)

                if self.method == "wrap":
                    # From (Euclidean) tangent space at origin to Poincare ball
                    leaf_poin = self.project_t02p(z_leaf, q_leaves, get_jacob=False)
                    if self.connect_method == 'mst':
                        int_poin = self.project_t02p(z_int, q_ints, get_jacob=False)

                elif self.method == "logit":
                    sigmoid_transformation = SigmoidTransform()

                    # transformation of z from R^n to unit ball P^n
                    leaf_poin = sigmoid_transformation(z_leaf) * 2 - 1
                    if self.connect_method == 'mst':
                        int_poin = sigmoid_transformation(z_int) * 2 - 1

                # Change coordinates
                leaf_r, leaf_dir = utilFunc.cart_to_dir(leaf_poin.reshape((self.S, self.D)))
                if self.connect_method == 'mst':
                    int_r, int_dir = utilFunc.cart_to_dir(int_poin.reshape((self.S - 2, self.D)))

                # prepare return (peel, branch lengths, locations, and log posteriori)
                if self.connect_method == 'mst':
                    pl = utilFunc.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
                elif self.connect_method == 'geodesics':
                    pl, int_poin = utilFunc.make_peel_geodesics(leaf_poin.reshape((self.S, self.D)))
                    int_r, int_dir = utilFunc.cart_to_dir(int_poin.reshape((self.S - 1, self.D)))
                peel.append(pl)
                bl = self.compute_branch_lengths(self.S, pl, leaf_r, leaf_dir, int_r, int_dir)
                blens.append(bl)
                location.append(utilFunc.dir_to_cart_tree(leaf_r, int_r, leaf_dir, int_dir, self.D))
                if kwargs.get('lp'):
                    lp.append(calculate_treelikelihood(
                        self.partials, self.weights, pl, JC69_p_t(bl), torch.full([4], 0.25, dtype=torch.float64)))

            if kwargs.get('lp'):
                return peel, blens, location, lp
            else:
                return peel, blens, location

    def calculate_elbo(self, q_leaves, q_ints):
        """Calculate the elbo of a sample from the variational distributions q

        Args:
            q_leaf (Multivariate distribution):
                Distributions of leave locations in tangent space of hyperboloid T_0 H^n
            q_int (Multivariate distribution):
                Distributions of internal node locations in tangent space of hyperboloid T_0 H^n

        Returns:
            float: The evidence lower bound of a sample from q
        """
        # Normalise weights
        with torch.no_grad():
            self.VariationalParams['leaf_weights'] = self.VariationalParams['leaf_weights'] / torch.sum(
                self.VariationalParams['leaf_weights'])
            if self.connect_method == 'mst':
                self.VariationalParams['int_weights'] = self.VariationalParams['int_weights'] / torch.sum(
                    self.VariationalParams['int_weights'])

        # Sample in tangent space
        z_leaf, logQ = self.sample_loc(q_leaves)
        if self.connect_method == 'mst':
            z_int, logQ_int = self.sample_loc(q_ints)
            logQ += logQ_int

        if self.method == "wrap":
            # From (Euclidean) tangent space at origin to Poincare ball
            leaf_poin, log_abs_det_jacobian = self.project_t02p(z_leaf, q_leaves)
            if self.connect_method == 'mst':
                int_poin, int_jacobian = self.project_t02p(z_int, q_ints)
                log_abs_det_jacobian += int_jacobian

        elif self.method == "logit":
            sigmoid_transformation = SigmoidTransform()

            # transformation of z from R^n to unit ball P^n
            leaf_poin = sigmoid_transformation(z_leaf) * 2 - 1
            # take transformations into account
            log_abs_det_jacobian = sigmoid_transformation.log_abs_det_jacobian(leaf_poin, z_leaf).sum()
            if self.connect_method == 'mst':
                int_poin = sigmoid_transformation(z_int) * 2 - 1
                log_abs_det_jacobian += sigmoid_transformation.log_abs_det_jacobian(int_poin, z_int).sum()

        # Change leaf coordinates
        leaf_r, leaf_dir = utilFunc.cart_to_dir(leaf_poin.reshape((self.S, self.D)))

        # make_peel for prior and likelihood
        if self.connect_method == 'mst':
            int_r, int_dir = utilFunc.cart_to_dir(int_poin.reshape((self.S - 2, self.D)))
            peel = utilFunc.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
        elif self.connect_method == 'geodesics':
            peel, int_poin = utilFunc.make_peel_geodesics(leaf_poin.reshape((self.S, self.D)))
            int_r, int_dir = utilFunc.cart_to_dir(int_poin.reshape((self.S - 1, self.D)))
        blen = self.compute_branch_lengths(self.S, peel, leaf_r, leaf_dir, int_r, int_dir)

        # logPrior
        logPrior = self.compute_prior(peel, blen, **self.prior)

        # Likelihood
        logP = self.compute_LL(peel, blen)

        return logP + logPrior - logQ + log_abs_det_jacobian

    def project_t02p(self, z, q, get_jacob=True):
        # Project sample z (from distribution q) from tangent plane to poincare disc
        # Return: leaf_r, leaf_dir, log_abs_det_jacobian
        n_points = int(len(z)/self.D)
        mu = torch.zeros_like(z)
        for k in range(self.boosts):
            mu = mu + self.VariationalParams['leaf_weights'][k] * q[k].loc
        z_poin = t02p(z, self.D, mu).reshape(n_points, self.D)

        if not get_jacob:
            return z_poin
        else:
            # Get Jacobians
            log_abs_det_jacobian = torch.zeros(1)
            D = torch.tensor(self.D, dtype=float)

            # Leaves
            # Jacobian of t02p going from Tangent T_0 to Poincare ball
            J = torch.autograd.functional.jacobian(t02p, (z, D, mu), vectorize=True)
            J = J[0].reshape((n_points * self.D, n_points * self.D))
            log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(torch.det(J)))

            return z_poin, log_abs_det_jacobian

    def learn(self, param_init=None, epochs=1000, k_samples=3, path_write='./out', lr=1e-3):
        """Learn the variational parameters using Adam optimiser
        Args:
            param_init (dict, optional): Initial parameters. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 1000.
            k_samples (int, optional): Number of tree samples at each epoch. Defaults to 3.
        """
        print("Using %i tree samples at each epoch." % k_samples)
        print("Running for %i epochs.\n" % epochs)

        if path_write is not None:
            fn = path_write + '/' + 'vi.info'
            with open(fn, 'w') as file:
                file.write('%-12s: %i\n' % ("# epochs", epochs))
                file.write('%-12s: %i\n' % ("k_samples", k_samples))
                file.write('%-12s: %i\n' % ("Dimensions", self.D))
                file.write('%-12s: %i\n' % ("# Taxa", self.S))
                file.write('%-12s: %i\n' % ("Patterns", self.L))
                file.write('%-12s: %i\n' % ("Boosts", self.boosts))
                file.write('%-12s: %f\n' % ("Learn Rate", lr))
                file.write('%-12s: %s\n' % ("Embed Mthd", self.method))
                file.write('%-12s: %s\n' % ("Connect Mthd", self.connect_method))
                for key, value in self.prior.items():
                    file.write('%-12s: %f\n' % (key, value))

        if param_init is not None:
            self.VariationalParams["leaf_mu"] = param_init["leaf_mu"]
            self.VariationalParams["leaf_sigma"] = param_init["leaf_sigma"]
            self.VariationalParams["leaf_weights"] = torch.full((self.boosts, 1), 1/self.boosts).requires_grad_(True)
            if self.connect_method == 'mst':
                self.VariationalParams["int_mu"] = param_init["int_mu"]
                self.VariationalParams["int_sigma"] = param_init["int_sigma"]
                self.VariationalParams["int_weights"] = torch.full((self.boosts, 1), 1/self.boosts).requires_grad_(True)

            # Save varitaional parameters
            fn = os.path.join(path_write, "VI_params_init.csv")
            self.save_VI(fn)

        lr_lambda = lambda epoch: 1.0 / np.sqrt(epoch + 1)
        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = - self.elbo_normal(k_samples)
            if loss.requires_grad:
                loss.backward()
            elbo_hist.append(- loss.item())
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            print('epoch %-12i ELBO: %10.3f' % (epoch+1, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

        if epochs > 0 and path_write is not None:
            plt.figure()
            plt.plot(range(epochs), elbo_hist, 'r', label='elbo')
            plt.title('Elbo values')
            plt.xlabel('Epochs')
            plt.ylabel('elbo')
            plt.legend()
            plt.savefig(path_write + "/elbo_trace.png")

            plt.clf()
            plt.hist(hist_dat)
            plt.savefig(path_write + "/elbo_hist.png")

            fn = os.path.join(path_write, 'elbo.txt')
            with open(fn, 'w') as f:
                for i in range(epochs):
                    f.write("%f\n" % elbo_hist[i])

        final_elbo = self.elbo_normal(100).item()
        print('Final ELBO: {}'.format(final_elbo))
        if path_write is not None:
            fn = os.path.join(path_write, 'vi.info')
            with open(fn, 'a') as file:
                file.write('%-12s: %i\n' % ("Final ELBO", final_elbo))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        # q_thetas in tangent space at origin in T_0 H^dim. Each point i, has a multivariate normal in dim=D
        q_leaf = []
        q_int = []
        for k in range(self.boosts):
            mu = self.VariationalParams["leaf_mu"][k].reshape(self.S * self.D)
            cov = self.VariationalParams["leaf_sigma"][k].exp().reshape(self.S * self.D) * torch.eye(self.S * self.D)
            q_leaf.append(MultivariateNormal(mu, cov))

            if self.connect_method == 'mst':
                mu = self.VariationalParams["int_mu"][k].reshape((self.S - 2) * self.D)
                cov = self.VariationalParams["int_sigma"][k].exp().reshape((self.S - 2) * self.D)\
                    * torch.eye((self.S - 2) * self.D)
                q_int.append(MultivariateNormal(mu, cov))

        elbos = []
        for _ in range(size):
            elbos.append(self.calculate_elbo(q_leaf, q_int))
        return torch.mean(torch.stack(elbos))

    @staticmethod
    def run(dim, S, partials, weights, dists, path_write,
            epochs=1000, k_samples=3, n_draws=100, boosts=1,
            init_grids=10, init_trials=100, method='wrap', lr=1e-3,
            connect_method='mst', **prior):
        """Initialise and run Dodonaphy's variational inference

        Initialise the emebedding with tips distances given to hydra.
        Internal nodes are in distributions at origin.

        """
        print('\nRunning Dodonaphy Variational Inference.')
        print('Using %s embedding with %s connections' % (method, connect_method))

        # embed tips with hydra
        emm_tips = utilFunc.hydra(D=dists, dim=dim, equi_adj=0., stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyVI(partials, weights, dim, boosts, method=method, connect_method=connect_method, **prior)

        # Choose internal node locations from best random initialisation
        if connect_method == 'mst':
            int_r, int_dir = mymod.initialise_ints(emm_tips, n_grids=init_grids, n_trials=init_trials)

        # convert to tangent space
        leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm_tips["r"]), torch.from_numpy(emm_tips["directional"]))
        if connect_method == 'mst':
            int_loc_poin = torch.from_numpy(utilFunc.dir_to_cart(int_r, int_dir))

        if method == 'wrap':
            leaf_loc_t0 = p2t0(leaf_loc_poin).detach().numpy()
            if connect_method == 'mst':
                int_loc_t0 = p2t0(int_loc_poin).detach().numpy()
        elif method == 'logit':
            leaf_loc_poin = (leaf_loc_poin + 1)/2
            leaf_loc_t0 = np.log(leaf_loc_poin/(1-leaf_loc_poin))
            if connect_method == 'mst':
                int_loc_poin = (int_loc_poin + 1)/2
                int_loc_t0 = np.log(int_loc_poin/(1-int_loc_poin))

        # set variational parameters with small coefficient of variation
        cv = 1. / 50
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
        if connect_method == 'mst':
            int_sigma = np.log(np.abs(np.array(int_loc_t0)) * cv + eps)

        leaf_loc_t0_b = DodonaphyVI.init_boosters(leaf_loc_t0, boosts)
        leaf_sigma_b = DodonaphyVI.init_boosters(leaf_sigma, boosts)
        if connect_method == 'mst':
            int_loc_t0_b = DodonaphyVI.init_boosters(int_loc_t0, boosts)
            int_sigma_b = DodonaphyVI.init_boosters(int_sigma, boosts)

        if connect_method == 'mst':
            param_init = {
                "leaf_mu": torch.tensor(leaf_loc_t0_b, requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma_b, requires_grad=True, dtype=torch.float64),
                "int_mu": torch.tensor(int_loc_t0_b, requires_grad=True, dtype=torch.float64),
                "int_sigma": torch.tensor(int_sigma_b, requires_grad=True, dtype=torch.float64)
            }
        elif connect_method == 'geodesics':
            param_init = {
                "leaf_mu": torch.tensor(leaf_loc_t0_b, requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma_b, requires_grad=True, dtype=torch.float64)
            }

        # learn
        mymod.learn(
            param_init=param_init,
            epochs=epochs,
            k_samples=k_samples,
            path_write=path_write,
            lr=lr)

        # draw samples
        peels, blens, X, lp = mymod.draw_sample(n_draws, lp=True)

        utilFunc.save_tree_head(path_write, "vi", S)
        for i in range(n_draws):
            utilFunc.save_tree(path_write, "vi", peels[i], blens[i], i, lp[i].item())

        # Save varitaional parameters
        fn = os.path.join(path_write, "VI_params.csv")
        mymod.save_VI(fn)

    def save_VI(self, fn):
        with open(fn, 'w') as f:
            for b in range(self.boosts):
                for i in range(self.S):
                    for d in range(self.D):
                        f.write('%f\t' % self.VariationalParams['leaf_mu'][b][i, d])
                f.write('\n')

            for b in range(self.boosts):
                for i in range(self.S):
                    for d in range(self.D):
                        f.write('%f\t' % self.VariationalParams['leaf_sigma'][b][i, d])
                f.write('\n')

            for i in range(self.boosts):
                f.write('%f\t' % self.VariationalParams['leaf_weights'][b])
            f.write('\n')

            if self.connect_method == 'mst':
                for b in range(self.boosts):
                    for i in range(self.S-2):
                        for d in range(self.D):
                            f.write('%f\t' % self.VariationalParams['int_mu'][b][i, d])
                    f.write('\n')

                for b in range(self.boosts):
                    for i in range(self.S-2):
                        for d in range(self.D):
                            f.write('%f\t' % self.VariationalParams['int_sigma'][b][i, d])
                    f.write('\n')

                for b in range(self.boosts):
                    f.write('%f\t' % self.VariationalParams['int_weights'][b])
                f.write('\n')


def read_VI(path_read, connect_method='mst'):
    with open(path_read, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    VariationalParams = {}
    VariationalParams['leaf_mu'] = np.array([float(i) for i in lines[0].rstrip().split("\t")])
    VariationalParams['leaf_sigma'] = np.array([float(i) for i in lines[1].rstrip().split("\t")])
    VariationalParams['leaf_weights'] = np.array([float(i) for i in lines[2].rstrip().split("\t")])
    if connect_method == 'mst':
        VariationalParams['int_mu'] = np.array([float(i) for i in lines[3].rstrip().split("\t")])
        VariationalParams['int_sigma'] = np.array([float(i) for i in lines[4].rstrip().split("\t")])
        VariationalParams['int_weights'] = np.array([float(i) for i in lines[5].rstrip().split("\t")])
    return VariationalParams
