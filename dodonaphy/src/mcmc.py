from .utils import utilFunc
from .hyperboloid import t02p, p2t0
from .base_model import BaseModel

from torch.distributions import normal, uniform
import torch
import numpy as np


class DodonaphyMCMC(BaseModel):

    def __init__(self, partials, weights, dim, loc=None, **prior):
        super().__init__(partials, weights, dim, **prior)
        self.loc = loc

    def learn(self, epochs, burnin=0, path_write='./out', save_period=1, step_scale=0.01):
        print("Running for %i epochs.\n" % epochs)
        self.step_scale = step_scale
        self.save_period = save_period

        fn = path_write + '/' + 'mcmc.info'
        with open(fn, 'w') as file:
            file.write('%-12s: %i\n' % ("# epochs", epochs))
            file.write('%-12s: %i\n' % ("Burnin", burnin))
            file.write('%-12s: %i\n' % ("Save period", save_period))
            file.write('%-12s: %f\n' % ("Step Scale", step_scale))
            file.write('%-12s: %i\n' % ("Dimensions", self.D))
            file.write('%-12s: %i\n' % ("# Taxa", self.S))
            file.write('%-12s: %i\n' % ("Unique sites", self.L))
            for key, value in self.prior.items():
                file.write('%-12s: %f\n' % (key, value))

        utilFunc.save_tree_head(path_write, "mcmc", self.S)

        for _ in range(burnin):
            self.evolove()

        accepted = 0

        for i in range(epochs+1):
            # set peel + blens + poincare locations
            loc_vec = self.loc.reshape(self.bcount * self.D)
            loc_poin = t02p(loc_vec, torch.zeros_like(loc_vec), self.D).reshape(self.bcount, self.D)
            leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_poin)
            self.peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
            self.blens = self.compute_branch_lengths(self.S, self.D, self.peel, leaf_r, leaf_dir, int_r, int_dir)
            loc_poin = torch.cat((loc_poin, torch.unsqueeze(loc_poin[0, :], axis=0)))

            # save
            if i % self.save_period == 0:
                if i > 0:
                    print('epoch: %-12i Acceptance Rate: %5.3f' % (i, accepted/i))
                else:
                    self.lnP = self.compute_LL(self.peel, self.blens)
                utilFunc.save_tree(path_write, 'mcmc', self.peel, self.blens, i*self.bcount, self.lnP)
                fn = path_write + '/mcmc_locations.csv'
                with open(fn, 'a') as file:
                    file.write(
                        np.array2string(loc_vec.data.numpy())
                        .replace('\n', '').replace('[', '').replace(']', '') + "\n")

            # step
            if i < epochs:
                accepted += self.evolve()

        fn = path_write + '/' + 'mcmc.info'
        with open(fn, 'a') as file:
            file.write('%-12s: %f\n' % ("Acceptance", accepted/epochs))

    def evolve(self):
        accepted = 0
        for i in range(self.bcount):
            loc_proposal = self.loc.detach().clone()
            loc_proposal[i, :] = loc_proposal[i, :] + normal.Normal(0, self.step_scale).sample((1, self.D))
            r, like_proposal = self.accept_ratio(loc_proposal)

            accept = False
            if r >= 1:
                accept = True
            elif uniform.Uniform(torch.zeros(1), torch.ones(1)).sample() < r:
                accept = True

            if accept:
                self.loc = loc_proposal
                self.lnP = like_proposal
                accepted += 1

        return accept

    def accept_ratio(self, loc_proposal):

        # current likelihood + prior
        current_like = self.compute_LL(self.peel, self.blen)
        current_prior = self.compute_prior(self.peel, self.blen, **self.prior)

        # proposal likelihood + prior
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_proposal)
        peel = utilFunc.make_peel(leaf_r, int_r, leaf_dir, int_dir)
        blen = self.compute_branch_lengths(self.S, self.D, peel, leaf_r, int_r, leaf_dir, int_dir)
        prop_like = self.compute_LL(peel, blen)
        prop_prior = self.compute_prior(peel, blen, **self.prior)

        # likelihood ratio
        like_ratio = torch.exp(prop_like - current_like)

        # prior ratio
        prior_ratio = prop_prior / current_prior

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        return torch.minimum(torch.ones(1), prior_ratio * like_ratio * hastings_ratio), prop_like

    @staticmethod
    def run(dim, partials, weights, dists, path_write,
            epochs=1000, step_scale=0.01, save_period=1,
            n_grids=10, n_trials=100, **prior):
        print('\nRunning Dodonaphy MCMC')

        # embed tips with distances using Hydra
        emm = utilFunc.hydra(dists, dim=dim, equi_adj=0.0, stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(emm["stress"].item()))

        # Initialise model
        mymod = DodonaphyMCMC(partials, weights, dim, **prior)

        # Choose internal node locations from best random initialisation
        # TODO: use non-static method. When to initialise self.loc??
        int_r, int_dir = mymod.initialise_ints(emm, n_grids=n_grids, n_trials=n_trials, max_scale=5)
        emm["r"] = np.concatenate((emm["r"], int_r))
        emm["directional"] = np.concatenate((emm["directional"], int_dir))

        # store in tangent plane R^dim
        loc_poin = utilFunc.dir_to_cart(torch.from_numpy(emm["r"]), torch.from_numpy(emm["directional"]))
        loc_t0 = p2t0(loc_poin)
        mymod.loc = loc_t0

        # Learn
        mymod.learn(epochs, path_write=path_write, step_scale=step_scale, save_period=save_period)
