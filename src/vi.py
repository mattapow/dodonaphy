from typing import List, Any

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .hyperboloid import t02p
from .phylo import calculate_treelikelihood, JC69_p_t
from .utils import utilFunc
from .base_model import BaseModel
from src.hyperboloid import p2t0
import matplotlib.pyplot as plt
import matplotlib.cm


class DodonaphyVI(BaseModel):

    def __init__(self, partials, weights, dim, **prior):
        super().__init__(partials, weights, dim, **prior)

        # Store mu on poincare ball in R^dim.
        # Distributions stored in tangent space T_0 H^D, then transformed to poincare ball.
        # The distribution for each point has a single sigma (i.e. mean field in x, y).
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.random.randn(self.S, self.D)) + eps)
        int_sigma = np.log(np.abs(np.random.randn(self.S - 2, self.D)) + eps)
        self.VariationalParams = {
            "leaf_mu": torch.randn((self.S, self.D), requires_grad=True, dtype=torch.float64),
            "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
            "int_mu": torch.randn((self.S - 2, self.D), requires_grad=True, dtype=torch.float64),
            "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64)
        }

    def sample_loc(self, q_leaf, q_int, get_z=False):
        # Sample z in tangent space of hyperboloid at origin T_0 H^n
        z_leaf = q_leaf.rsample((1,)).squeeze()
        z_int = q_int.rsample((1,)).squeeze()

        # From (Euclidean) tangent space at origin to Poincare ball
        mu_leaf = q_leaf.loc
        mu_int = q_int.loc

        # Tangent space at origin to Poincare
        z_leaf_poin = t02p(z_leaf, mu_leaf, self.D).reshape(self.S, self.D)
        z_int_poin = t02p(z_int, mu_int, self.D).reshape(self.S-2, self.D)

        # transform z to r, dir
        leaf_r, leaf_dir = utilFunc.cart_to_dir(z_leaf_poin)
        int_r, int_dir = utilFunc.cart_to_dir(z_int_poin)

        if get_z:
            return leaf_r, leaf_dir, int_r, int_dir, z_leaf, mu_leaf, z_int, mu_int
        else:
            return leaf_r, leaf_dir, int_r, int_dir

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
            # q_thetas in tangent space at origin in T_0 H^dim.
            mu = self.VariationalParams["leaf_mu"].reshape(self.S * self.D)
            cov = self.VariationalParams["leaf_sigma"].exp().reshape(self.S * self.D) * torch.eye(self.S * self.D)
            q_leaf = MultivariateNormal(mu, cov)

            mu = self.VariationalParams["int_mu"].reshape((self.S - 2) * self.D)
            cov = self.VariationalParams["int_sigma"].exp().reshape((self.S - 2) * self.D)\
                * torch.eye((self.S - 2) * self.D)
            q_int = MultivariateNormal(mu, cov)

            # make peel, blens and X for each of these samples
            peel = []
            blens = []
            location = []
            lp = []
            for _ in range(nSample):
                leaf_r, leaf_dir, int_r, int_dir = self.sample_loc(q_leaf, q_int, get_z=False)

                # prepare return (peel, branch lengths, locations, and log posteriori)
                pl = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
                peel.append(pl)
                bl = self.compute_branch_lengths(self.S, self.D, pl, leaf_r, leaf_dir, int_r, int_dir)
                blens.append(bl)
                location.append(utilFunc.dir_to_cart_tree(leaf_r, int_r, leaf_dir, int_dir, self.D))
                if kwargs.get('lp'):
                    lp.append(calculate_treelikelihood(
                        self.partials, self.weights, pl, JC69_p_t(bl), torch.full([4], 0.25, dtype=torch.float64)))

            if kwargs.get('lp'):
                return peel, blens, location, lp
            else:
                return peel, blens, location

    def calculate_elbo(self, q_leaf, q_int):
        """Calculate the elbo of a sample from the variational distributions q

        Args:
            q_leaf (Multivariate distribution):
                Distributions of leave locations in tangent space of hyperboloid T_0 H^n
            q_int (Multivariate distribution):
                Distributions of internal node locations in tangent space of hyperboloid T_0 H^n

        Returns:
            float: The evidence lower bound of a sample from q
        """
        leaf_r, leaf_dir, int_r, int_dir, z_leaf, mu_leaf, z_int, mu_int = self.sample_loc(q_leaf, q_int, get_z=True)

        # Get Jacobians
        log_abs_det_jacobian = torch.zeros(1)
        D = torch.tensor(self.D, dtype=float)

        # Leaves
        # Jacobian of t02p going from Tangent T_0 to Poincare ball
        J_leaf = torch.autograd.functional.jacobian(t02p, (z_leaf, mu_leaf, D))
        J_leaf = J_leaf[0].reshape((self.S * self.D, self.S * self.D))
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(torch.det(J_leaf)))
        # Jacobian of going to polar
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(1/leaf_r).sum(0)

        # Internal nodes
        J_int = torch.autograd.functional.jacobian(t02p, (z_int, mu_int, D))
        J_int = J_int[0].reshape(((self.S - 2) * self.D, (self.S - 2) * self.D))
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(torch.det(J_int)))
        log_abs_det_jacobian = log_abs_det_jacobian + torch.log(1 / int_r).sum(0)

        # logQ
        logQ = 0
        logQ = logQ + q_leaf.log_prob(z_leaf)
        logQ = logQ + q_int.log_prob(z_int)

        # logPrior
        logPrior = torch.tensor(self.compute_prior(
            leaf_r, leaf_dir, int_r, int_dir, **self.prior), requires_grad=False)

        logP = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)

        return logP + logPrior - logQ + log_abs_det_jacobian

    def learn(self, param_init=None, epochs=1000, k_samples=3, path_write='./out'):
        """Learn the variational parameters using Adam optimiser
        Args:
            param_init (dict, optional): Initial parameters. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 1000.
            k_samples (int, optional): Number of tree samples at each epoch. Defaults to 3.
        """
        print("Using %i tree samples at each epoch." % k_samples)
        print("Running for %i epochs.\n" % epochs)

        if not path_write == "":
            fn = path_write + '/' + 'vi.info'
            with open(fn, 'w') as file:
                file.write('%-12s: %i\n' % ("# epochs", epochs))
                file.write('%-12s: %i\n' % ("k_samples", k_samples))
                file.write('%-12s: %i\n' % ("Dimensions", self.D))
                file.write('%-12s: %i\n' % ("# Taxa", self.S))
                file.write('%-12s: %i\n' % ("Unique sites", self.L))
                for key, value in self.prior.items():
                    file.write('%-12s: %f\n' % (key, value))

        if param_init is not None:
            # set initial params as a Dict
            self.VariationalParams["leaf_mu"] = param_init["leaf_mu"]
            self.VariationalParams["leaf_sigma"] = param_init["leaf_sigma"]
            self.VariationalParams["int_mu"] = param_init["int_mu"]
            self.VariationalParams["int_sigma"] = param_init["int_sigma"]

        lr_lambda = lambda epoch: 1.0 / np.sqrt(epoch + 1)
        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []
        for epoch in range(epochs):
            loss = - self.elbo_normal(k_samples)
            elbo_hist.append(- loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print('epoch %-12i ELBO: %10.3f' % (epoch+1, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

        if epochs > 0 and not path_write == "":
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

        print('Final ELBO: {}'.format(self.elbo_normal(100).item()))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        # q_thetas in tangent space at origin in T_0 H^dim. Each point i, has a multivariate normal in dim=D
        mu = self.VariationalParams["leaf_mu"].reshape(self.S * self.D)
        cov = self.VariationalParams["leaf_sigma"].exp().reshape(self.S * self.D) * torch.eye(self.S * self.D)
        q_leaf = MultivariateNormal(mu, cov)

        mu = self.VariationalParams["int_mu"].reshape((self.S - 2) * self.D)
        cov = self.VariationalParams["int_sigma"].exp().reshape((self.S - 2) * self.D)\
            * torch.eye((self.S - 2) * self.D)
        q_int = MultivariateNormal(mu, cov)

        elbos = []
        for _ in range(size):
            elbos.append(self.calculate_elbo(q_leaf, q_int))
        return torch.mean(torch.stack(elbos))

    @staticmethod
    def run_tips(dim, S, partials, weights, dists, path_write, epochs=1000, k_samples=3, n_draws=100, **prior):
        """Initialise and run Dodonaphy's variational inference

        Initialise the emebedding with tips distances given to hydra.
        Internal nodes are in distributions at origin.

        """
        print('\nRunning Dodonaphy Variational Inference')

        # embed tips with hydra
        emm_tips = utilFunc.hydra(D=dists, dim=dim, equi_adj=0., stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyVI(partials, weights, dim, **prior)

        # Choose internal node locations from best random initialisation
        int_r, int_dir = mymod.initialise_ints(emm_tips, n_scale=10, n_trials=100, max_scale=5)

        # convert to tangent space
        leaf_loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm_tips["r"]), torch.from_numpy(emm_tips["directional"]))
        leaf_loc_t0 = p2t0(leaf_loc_poin).detach().numpy()
        int_loc_poin = torch.from_numpy(utilFunc.dir_to_cart(int_r, int_dir))
        int_loc_t0 = p2t0(int_loc_poin).detach().numpy()

        # set variational parameters with small coefficient of variation
        cv = 1. / 50
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.array(leaf_loc_t0)) * cv + eps)
        int_sigma = np.log(np.abs(np.array(int_loc_t0)) * cv + eps)
        param_init = {
            "leaf_mu": torch.tensor(leaf_loc_t0, requires_grad=True, dtype=torch.float64),
            "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
            "int_mu": torch.tensor(int_loc_t0, requires_grad=True, dtype=torch.float64),
            "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64)
        }

        # learn
        mymod.learn(param_init=param_init, epochs=epochs, k_samples=k_samples, path_write='./out')

        # draw samples
        peels, blens, X, lp = mymod.draw_sample(n_draws, lp=True)

        # Plot embedding if dim==2
        if dim == 2:
            _, ax = plt.subplots(1, 1)
            ax.set(xlim=[-1, 1])
            ax.set(ylim=[-1, 1])
            cmap = matplotlib.cm.get_cmap('Spectral')
            n_plots = min(n_draws, 10)
            for i in range(n_plots-1):
                utilFunc.plot_tree(ax, peels[i], X[i].detach().numpy(), color=cmap(i / n_draws), labels=False)
            utilFunc.plot_tree(ax, peels[i+1], X[i+1].detach().numpy(), color=cmap((i+1) / n_draws), labels=True)
            ax.set_title("Final Embedding Sample")
            plt.clf()
            plt.savefig("./out/vi_embeddings.png")

        utilFunc.save_tree_head(path_write, "vi", S)
        for i in range(n_draws):
            utilFunc.save_tree(path_write, "vi", peels[i], blens[i], i, lp[i].item())
