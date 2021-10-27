import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import hydra, tree, utils, Cutils
from .base_model import BaseModel
from .hyperboloid import p2t0
from .phylo import JC69_p_t, calculate_treelikelihood


class DodonaphyVI(BaseModel):
    def __init__(self, partials, weights, dim, embed_method='wrap', connect_method='mst', curvature=-1.,
                 dists_data=None, **prior):
        super().__init__(partials, weights, dim, connect_method=connect_method, curvature=curvature,
                         dists_data=dists_data, **prior)
        print('Initialising variational model.\n')

        # Store mu on poincare ball in R^dim.
        # Distributions stored in tangent space T_0 H^D, then transformed to poincare ball.
        # The distribution for each point has a single sigma (i.e. mean field in x, y).
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.random.randn(self.S, self.D)) + eps)
        int_sigma = np.log(np.abs(np.random.randn(self.S - 2, self.D)) + eps)
        assert embed_method in ('wrap', 'simple')
        self.embed_method = embed_method
        assert connect_method in ('mst', 'geodesics', 'incentre', 'nj')
        self.connect_method = connect_method

        if self.connect_method in ('geodesics', 'incentre', 'nj'):
            self.VariationalParams = {
                "leaf_mu": torch.randn((self.S, self.D), requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
            }
        elif self.connect_method == 'mst':
            self.VariationalParams = {
                "leaf_mu": torch.randn((self.S, self.D), requires_grad=True, dtype=torch.float64),
                "leaf_sigma": torch.tensor(leaf_sigma, requires_grad=True, dtype=torch.float64),
                "int_mu": torch.randn((self.S-2, self.D), requires_grad=True, dtype=torch.float64),
                "int_sigma": torch.tensor(int_sigma, requires_grad=True, dtype=torch.float64),
            }

    # leaf_dir = np.randn((self.S, self.D))
    # leaf_dir = leaf_dir / np.pow(np.sum(leaf_dir**2, dim=-1), .5).repeat(1, self.D)
    # leaf_dir_sigma = np.abs(np.random.randn(self.S, self.D))

    # int_r_sigma = np.abs(np.random.randn(self.S))
    # int_dir = np.randn((self.S - 2, self.D))
    # int_dir = int_dir / np.pow(np.sum(int_dir**2, dim=-1), .5).repeat(1, self.D)
    # int_dir_sigma = np.abs(np.random.randn(self.S - 2, self.D))

    # if self.connect_method == 'geodesics' or self.connect_method == 'incentre':
    #     self.VariationalParams = {
    #         "leaf_r": torch.randn((1), requires_grad=True, dtype=torch.float64),
    #         "leaf_r_sigma": torch.tensor(.1, requires_grad=True, dtype=torch.float64),
    #         "leaf_dir": torch.tensor(leaf_dir, requires_grad=True, dtype=torch.float64),
    #         "leaf_dir_sigma": torch.tensor(leaf_dir_sigma, requires_grad=True, dtype=torch.float64)
    #     }
    # elif self.connect_method == 'mst':
    #     self.VariationalParams = {
    #         "leaf_r": torch.randn((1), requires_grad=True, dtype=torch.float64),
    #         "leaf_r_sigma": torch.tensor(.1, requires_grad=True, dtype=torch.float64),
    #         "leaf_dir": torch.tensor(leaf_dir, requires_grad=True, dtype=torch.float64),
    #         "leaf_dir_sigma": torch.tensor(leaf_dir_sigma, requires_grad=True, dtype=torch.float64),

    #         "int_r_mu": torch.full((self.S-2), .1, requires_grad=True, dtype=torch.float64),
    #         "int_r_sigma": torch.tensor(int_r_sigma, requires_grad=True, dtype=torch.float64),
    #         "int_dir_mu": torch.tensor(int_dir, requires_grad=True, dtype=torch.float64),
    #         "int_dir_sigma": torch.tensor(int_dir_sigma, requires_grad=True, dtype=torch.float64),
    #     }

    def draw_sample(self, nSample=100, **kwargs):
        """Draw samples from the variational posterior distribution

        Args:
            nSample (int, optional): Number of samples to be drawn. Defaults to 100.

        Returns:
            tuple[list list list list]: peel, blens, location, lp. If kwarg 'lp' is passed.
            Locations are in Poincare disk. lp = log-probability
            tuple[list list list]: peel, blens, location, lp. Otherwise.
        """
        # make peel, blens and X for each of these samples
        peel = []
        blens = []
        location = []
        lp = []
        with torch.no_grad():
            for _ in range(nSample):

                n_tip_params = torch.numel(self.VariationalParams["leaf_mu"])
                leaf_loc = self.VariationalParams["leaf_mu"].reshape(
                    n_tip_params)
                leaf_cov = torch.eye(n_tip_params, dtype=torch.double) *\
                    self.VariationalParams["leaf_sigma"].exp().reshape(
                        n_tip_params)
                if self.connect_method == 'mst':
                    n_int_params = torch.numel(
                        self.VariationalParams["int_mu"])
                    int_loc = self.VariationalParams["int_mu"].reshape(
                        n_int_params)
                    int_cov = torch.eye(n_int_params, dtype=torch.double) *\
                        self.VariationalParams["int_sigma"].exp().reshape(
                            n_int_params)
                    sample = self.sample(leaf_loc, leaf_cov, int_loc, int_cov)
                else:
                    sample = self.sample(leaf_loc, leaf_cov)

                peel.append(sample['peel'])
                blens.append(sample['blens'])
                if self.connect_method == 'mst':
                    location.append(utils.dir_to_cart_tree(
                        sample['leaf_r'].repeat(self.S), sample['int_r'],
                        sample['leaf_dir'], sample['int_dir'], self.D))
                else:
                    location.append(utils.dir_to_cart(
                        sample['leaf_r'].repeat(self.S), sample['leaf_dir']))
                if kwargs.get('lp'):
                    LL = calculate_treelikelihood(
                        self.partials, self.weights, sample['peel'], JC69_p_t(
                            sample['blens']),
                        torch.full([4], 0.25, dtype=torch.float64))
                    lp.append(LL)

        if kwargs.get('lp'):
            return peel, blens, location, lp
        else:
            return peel, blens, location

    def calculate_elbo(self):
        """Calculate the elbo of a sample from the variational distributions q

        Args:
            q_leaf (Multivariate distribution):
                Distributions of leave locations in tangent space of hyperboloid T_0 H^n
            q_int (Multivariate distribution):
                Distributions of internal node locations in tangent space of hyperboloid T_0 H^n

        Returns:
            float: The evidence lower bound of a sample from q
        """
        n_tip_params = torch.numel(self.VariationalParams["leaf_mu"])
        leaf_loc = self.VariationalParams["leaf_mu"].reshape(n_tip_params)
        leaf_cov = torch.eye(n_tip_params, dtype=torch.double) *\
            self.VariationalParams["leaf_sigma"].exp().reshape(n_tip_params)
        if self.connect_method == 'mst':
            n_int_params = torch.numel(self.VariationalParams["int_mu"])
            int_loc = self.VariationalParams["int_mu"].reshape(n_int_params)
            int_cov = torch.eye(n_int_params, dtype=torch.double) *\
                self.VariationalParams["int_sigma"].exp().reshape(n_int_params)
            sample = self.sample(leaf_loc, leaf_cov, int_loc, int_cov)
        else:
            sample = self.sample(leaf_loc, leaf_cov)

        if self.connect_method == 'mst':
            pdm = Cutils.get_pdm_torch(sample['leaf_r'].repeat(self.S), sample['leaf_dir'],
                                       sample['int_r'], sample['int_dir'],
                                       curvature=self.curvature)
        else:
            pdm = Cutils.get_pdm_torch(sample['leaf_r'].repeat(self.S), sample['leaf_dir'],
                                       curvature=self.curvature)

        # logPrior = self.compute_prior_gamma_dir(sample['blens'])
        # logP = self.compute_LL(sample['peel'], sample['blens'])
        # logPrior = self.compute_prior_gamma_dir(pdm[:])

        # anneal_epoch = torch.tensor(100)
        # min_temp = torch.tensor(.1)
        # temp = torch.maximum(torch.exp(- self.epoch / anneal_epoch), min_temp)
        # logP = self.compute_log_a_like(pdm, temp=temp)

        return sample['lnP'] + sample['lnPrior'] - sample['logQ'] + sample['jacobian']

    def learn(self, param_init=None, epochs=1000, k_samples=3, path_write='./out', lr=1e-3):
        """Learn the variational parameters using Adam optimiser
        Args:
            param_init (dict, optional): Initial parameters. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 1000.
            k_samples (int, optional): Number of tree samples at each epoch. Defaults to 3.
        """
        print("Using %i tree samples at each epoch." % k_samples)
        print("Running for %i epochs.\n" % epochs)

        # initialise variational parameters if given
        if param_init is not None:
            self.VariationalParams["leaf_mu"] = param_init["leaf_mu"]
            self.VariationalParams["leaf_sigma"] = param_init["leaf_sigma"]
            if self.connect_method == 'mst':
                self.VariationalParams["int_mu"] = param_init["int_mu"]
                self.VariationalParams["int_sigma"] = param_init["int_sigma"]

        if path_write is not None:
            fn = path_write + '/' + 'vi.info'
            with open(fn, 'w') as file:
                file.write('%-12s: %i\n' % ("# epochs", epochs))
                file.write('%-12s: %i\n' % ("k_samples", k_samples))
                file.write('%-12s: %i\n' % ("Dimensions", self.D))
                file.write('%-12s: %i\n' % ("# Taxa", self.S))
                file.write('%-12s: %i\n' % ("Patterns", self.L))
                file.write('%-12s: %f\n' % ("Learn Rate", lr))
                file.write('%-12s: %s\n' % ("Embed Mthd", self.embed_method))
                file.write('%-12s: %s\n' %
                           ("Connect Mthd", self.connect_method))
                for key, value in self.prior.items():
                    file.write('%-12s: %s\n' % (key, str(value)))

                # Save varitaional parameters
                fn = os.path.join(path_write, "VI_params_init.csv")
                self.save(fn)

            elbo_fn = os.path.join(path_write, 'elbo.txt')

        def lr_lambda(epoch): return 1.0 / np.sqrt(epoch + 1)
        optimizer = torch.optim.Adam(
            list(self.VariationalParams.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

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

            if path_write is not None:
                with open(elbo_fn, 'a') as f:
                    f.write("%f\n" % elbo_hist[-1])

        if epochs > 0 and path_write is not None:
            try:
                plt.figure()
                plt.plot(range(1, epochs), elbo_hist[1:], 'r', label='elbo')
                plt.title('Elbo values')
                plt.xlabel('Epochs')
                plt.ylabel('elbo')
                plt.legend()
                plt.savefig(path_write + "/elbo_trace.png")

                plt.clf()
                plt.hist(hist_dat)
                plt.savefig(path_write + "/elbo_hist.png")
            except Exception:
                print("Could not generate and save elbo figures.")

        final_elbo = self.elbo_normal(100).item()
        print('Final ELBO: {}'.format(final_elbo))
        if path_write is not None:
            fn = os.path.join(path_write, 'vi.info')
            with open(fn, 'a') as file:
                file.write('%-12s: %i\n' %
                           ("Final ELBO (100 samples)", final_elbo))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        elbos = torch.zeros(size)
        for i in range(size):
            elbos[i] = self.calculate_elbo()
        return torch.mean(elbos)

    @staticmethod
    def run(dim, S, partials, weights, dists_data, path_write,
            epochs=1000, k_samples=3, n_draws=100,
            n_grids=10, n_trials=100, max_scale=1,
            embed_method='wrap', lr=1e-3, connect_method='nj',
            **prior):
        """Initialise and run Dodonaphy's variational inference

        Initialise the emebedding with tips distances given to hydra.
        Internal nodes are in distributions at origin.

        """
        print('\nRunning Dodonaphy Variational Inference.')
        print('Using %s embedding with %s connections' %
              (embed_method, connect_method))

        # embed tips with hydra
        emm_tips = hydra.hydra(D=dists_data, dim=dim, equi_adj=0., stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(
            emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyVI(partials, weights, dim, embed_method=embed_method, connect_method=connect_method,
                            dists_data=dists_data, **prior)

        # Choose internal node locations from best random initialisation
        if connect_method == 'mst':
            int_r, int_dir = mymod.initialise_ints(
                emm_tips, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale)

        # convert to cartesian coords
        leaf_loc_poin = utils.dir_to_cart(torch.from_numpy(
            emm_tips["r"]), torch.from_numpy(emm_tips["directional"]))
        if connect_method == 'mst':
            int_loc_poin = torch.from_numpy(utils.dir_to_cart(int_r, int_dir))

        # set variational parameters with small coefficient of variation
        cv = 1. / 100
        eps = np.finfo(np.double).eps
        # leaf_sigma = np.log(np.abs(np.array(leaf_loc_poin)) * cv + eps)
        if connect_method == 'mst':
            int_sigma = np.log(np.abs(np.array(int_loc_poin)) * cv + eps)

        # set leaf variational sigma using closest neighbour
        dists_data[dists_data == 0] = np.inf
        closest = dists_data.min(axis=0)
        closest = np.repeat([closest], dim, axis=0).transpose()
        leaf_sigma = np.log(np.abs(closest) * cv + eps)

        if connect_method == 'mst':
            param_init = {
                "leaf_mu": leaf_loc_poin.clone().double().requires_grad_(True),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double().requires_grad_(True),
                "int_mu": int_loc_poin.clone().double().requires_grad_(True),
                "int_sigma": torch.from_numpy(int_sigma).double().requires_grad_(True)
            }
        elif connect_method in ('geodesics', 'incentre', 'nj'):
            param_init = {
                "leaf_mu": leaf_loc_poin.clone().double().requires_grad_(True),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double().requires_grad_(True)
            }

        # learn ML
        # mymod.learn_ML_brute(param_init=param_init)

        # learn
        mymod.learn(
            param_init=param_init,
            epochs=epochs,
            k_samples=k_samples,
            path_write=path_write,
            lr=lr)

        # draw samples (one-by-one for reduced memory requirements)
        # and save them
        if path_write is not None:
            tree.save_tree_head(path_write, "vi", S)
            for i in range(n_draws):
                peels, blens, X, lp = mymod.draw_sample(1, lp=True)
                lnPr = mymod.compute_prior_gamma_dir(blens[0])
                tree.save_tree(
                    path_write, "vi", peels[0], blens[0], i, lp[0].item(), lnPr.item())

            # Save varitaional parameters
            fn = os.path.join(path_write, "VI_params.csv")
            mymod.save(fn)

    def save(self, fn):
        with open(fn, 'w') as f:
            for i in range(self.S):
                for d in range(self.D):
                    f.write('%f\t' % self.VariationalParams['leaf_mu'][i, d])
            f.write('\n')

            for i in range(self.S):
                for d in range(self.D):
                    f.write('%f\t' %
                            self.VariationalParams['leaf_sigma'][i, d])
            f.write('\n')

            if self.connect_method == 'mst':
                for i in range(self.S-2):
                    for d in range(self.D):
                        f.write('%f\t' %
                                self.VariationalParams['int_mu'][i, d])
                f.write('\n')

                for i in range(self.S-2):
                    for d in range(self.D):
                        f.write('%f\t' %
                                self.VariationalParams['int_sigma'][i, d])
                f.write('\n')


def read(path_read, connect_method='mst'):
    with open(path_read, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    VariationalParams = {}
    VariationalParams['leaf_mu'] = np.array(
        [float(i) for i in lines[0].rstrip().split("\t")])
    VariationalParams['leaf_sigma'] = np.array(
        [float(i) for i in lines[1].rstrip().split("\t")])
    if connect_method == 'mst':
        VariationalParams['int_mu'] = np.array(
            [float(i) for i in lines[2].rstrip().split("\t")])
        VariationalParams['int_sigma'] = np.array(
            [float(i) for i in lines[3].rstrip().split("\t")])
    return VariationalParams
