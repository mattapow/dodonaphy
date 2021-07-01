from . import utils, tree, hydra, peeler
from .hyperboloid import t02p, p2t0
from .base_model import BaseModel

from torch.distributions import normal, uniform
import torch
import numpy as np


class Chain(BaseModel):
    def __init__(self, partials, weights, dim, loc=None, step_scale=0.01, temp=1, target_acceptance=.234, **prior):
        super().__init__(partials, weights, dim, **prior)
        self.loc = loc  # location in the tangent space. Leaves then ints
        self.step_scale = step_scale
        self.temp = temp
        if loc is not None:
            self.n_points = len(self.loc)
        self.accepted = 0
        self.iterations = 0
        self.target_acceptance = target_acceptance
        self.converged = [False] * 100
        self.moreTune = True
        self.embed_mthd = 'wrap'

    def set_probability(self, connect_method, embed_method):
        # set peel + blens + poincare locations
        if connect_method == 'geodesics':
            self.loc_vec = self.loc.reshape(self.S * self.D)
            loc_poin = t02p(self.loc_vec, self.D).reshape(self.S, self.D)
            self.peel, int_locs = peeler.make_peel_geodesic(loc_poin)
            int_r, int_dir = utils.cart_to_dir(int_locs)
            leaf_r, leaf_dir = utils.cart_to_dir(loc_poin)
        elif connect_method == 'incentre':
            self.loc_vec = self.loc.reshape(self.S * self.D)
            loc_poin = t02p(self.loc_vec, self.D).reshape(self.S, self.D)
            self.peel, int_locs = peeler.make_peel_incentre(loc_poin)
            int_r, int_dir = utils.cart_to_dir(int_locs)
            leaf_r, leaf_dir = utils.cart_to_dir(loc_poin)
        elif connect_method == 'mst':
            self.loc_vec = self.loc.reshape(self.bcount * self.D)
            if embed_method == 'wrap':
                loc_poin = t02p(self.loc_vec, self.D).reshape(self.bcount, self.D)
            elif embed_method == 'sigmoid':
                loc_poin = self.loc / (1 + torch.pow(torch.sum(self.loc**2, axis=1, keepdim=True), .5).repeat(1, 2))
            leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(loc_poin)
            self.peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

        self.blens = self.compute_branch_lengths(self.S, self.peel, leaf_r, leaf_dir, int_r, int_dir)
        loc_poin = torch.cat((loc_poin, torch.unsqueeze(loc_poin[0, :], axis=0)))

        # current likelihood + prior
        self.lnP = self.compute_LL(self.peel, self.blens)
        self.lnPrior = self.compute_prior(self.peel, self.blens, **self.prior)

    def evolve(self, connect_method, embed_method):
        accepted = 0

        for i in range(self.n_points):
            loc_proposal = self.loc.detach().clone()
            loc_proposal[i, :] = loc_proposal[i, :] + normal.Normal(0, self.step_scale).sample((1, self.D))
            r, like_proposal = self.accept_ratio(loc_proposal, connect_method, embed_method)

            accept = False
            if r >= 1:
                accept = True
            elif uniform.Uniform(torch.zeros(1), torch.ones(1)).sample() < r:
                accept = True

            if accept:
                self.loc = loc_proposal
                self.lnP = like_proposal
                accepted += 1
                self.accepted += 1
        self.iterations += self.n_points

        return accept

    def accept_ratio(self, loc_proposal, connect_method, embed_method):
        """Acceptance critereon for Metropolis-Hastings

        Args:
            loc_proposal ([type]): Proposal location in R^n
            connect_method ([type]): Connection method for make_peel.
            embed_method ([type]): Embedding method: R^n to P^n

        Returns:
            tuple: (r, prop_like)
            The acceptance ratio r and the likelihood of the proposal.
        """
        # proposal likelihood + prior
        if connect_method == 'geodesics':
            loc_proposal_vec = loc_proposal.reshape(self.S * self.D)
            loc_proposal_poin = t02p(loc_proposal_vec, self.D).reshape(self.S, self.D)
            leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(loc_proposal_poin)
            peel, int_proposal_poin = peeler.make_peel_geodesic(loc_proposal_poin)
            int_r, int_dir = utils.cart_to_dir(int_proposal_poin)
            leaf_r, leaf_dir = utils.cart_to_dir(loc_proposal_poin)
        elif connect_method == 'incentre':
            loc_proposal_vec = loc_proposal.reshape(self.S * self.D)
            loc_proposal_poin = t02p(loc_proposal_vec, self.D).reshape(self.S, self.D)
            leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(loc_proposal_poin)
            peel, int_proposal_poin = peeler.make_peel_incentre(loc_proposal_poin)
            int_r, int_dir = utils.cart_to_dir(int_proposal_poin)
            leaf_r, leaf_dir = utils.cart_to_dir(loc_proposal_poin)
        elif connect_method == 'mst':
            if embed_method == 'wrap':
                loc_proposal_vec = loc_proposal.reshape(self.bcount * self.D)
                loc_proposal_poin = t02p(loc_proposal_vec, self.D).reshape(self.bcount, self.D)
            elif embed_method == 'sigmoid':
                loc_proposal_poin = loc_proposal / (1 + torch.pow(
                    torch.sum(self.loc**2, axis=1, keepdim=True), .5).repeat(1, 2))
            leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(loc_proposal_poin)
            leaf_r, int_r, leaf_dir, int_dir = utils.cart_to_dir_tree(loc_proposal_poin)
            peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
        blen = self.compute_branch_lengths(self.S, peel, leaf_r, leaf_dir, int_r, int_dir)
        prop_like = self.compute_LL(peel, blen)
        prop_prior = self.compute_prior(peel, blen, **self.prior)

        # likelihood ratio
        like_ratio = torch.exp(prop_like - self.lnP)

        # prior ratio
        prior_ratio = torch.exp(prop_prior - self.lnPrior)

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        return torch.minimum(torch.ones(1), (prior_ratio * like_ratio)**self.temp * hastings_ratio), prop_like

    def tune_step(self, tol=0.01):
        """Tune the acceptance rate. Simple Euler method.

        Args:
            tol (float, optional): Tolerance. Defaults to 0.01.
        """
        if not self.moreTune:
            return

        lr = 0.1
        eps = np.finfo(np.double).eps
        acceptance = self.accepted / self.iterations
        dy = acceptance - self.target_acceptance
        self.step_scale = max(self.step_scale + lr * dy, eps)

        # check convegence
        self.converged.pop()
        self.converged.insert(0, np.abs(dy) < tol)
        if all(self.converged):
            self.moreTune = False
            print("Step tuned to %f." % self.step_scale)


class DodonaphyMCMC():

    def __init__(self, partials, weights, dim, loc=None, step_scale=0.01, nChains=1,
                 connect_method='incentre', embed_method='wrap', **prior):
        self.nChains = nChains
        self.chain = []
        assert connect_method in ("incentre", "geodesics", "mst")
        self.connect_method = connect_method
        assert embed_method in ("wrap", "", "sigmoid")
        self.embed_method = embed_method
        dTemp = 0.1
        for i in range(nChains):
            temp = 1./(1+dTemp*i)
            self.chain.append(Chain(partials, weights, dim, loc, step_scale, temp, **prior))

    def learn(self, epochs, burnin=0, path_write='./out', save_period=1):
        print("Using 1 cold chain and %d hot chains." % int(self.nChains-1))
        self.save_period = save_period

        if path_write is not None:
            info_file = path_write + '/' + 'mcmc.info'
            self.save_info(info_file, epochs, burnin, save_period)
            tree.save_tree_head(path_write, "mcmc", self.chain[0].S)

        if burnin > 0:
            print('Burning in for %d iterations.' % burnin)
            deceile = 1
            for i in range(burnin):
                if i / burnin * 10 > deceile:
                    print("%d%% " % (deceile*10), end="", flush=True)
                    deceile += 1
                for c in range(self.nChains):

                    # Set prior and likelihood
                    self.chain[c].set_probability(self.connect_method, self.embed_method)

                    # step
                    self.chain[c].evolve(self.connect_method, self.embed_method)
                    self.chain[c].tune_step()

                    # tune step
                    if self.chain[c].moreTune:
                        self.chain[c].tune_step()

                # swap 2 chains
                if self.nChains > 1:
                    _ = self.swap()
            print("100%")

        swaps = 0
        print("Running for %d epochs.\n" % epochs)
        for i in range(epochs):
            for c in range(self.nChains):
                # Set prior and likelihood
                self.chain[c].set_probability(self.connect_method, self.embed_method)

                # step
                self.chain[c].evolve(self.connect_method, self.embed_method)

                # tune step
                if self.chain[c].moreTune:
                    self.chain[c].tune_step()

            # swap 2 chains
            if self.nChains > 1:
                swaps += self.swap()

            # save
            doSave = path_write is not None and self.save_period > 0 and i % self.save_period == 0
            if doSave:
                self.save_iteration(path_write, i)

        if path_write is not None:
            fn = path_write + '/' + 'mcmc.info'
            with open(fn, 'a') as file:
                final_accept = np.average(
                    [self.chain[c].accepted / self.chain[c].iterations for c in range(self.nChains)])
                file.write('%-12s: %f\n' % ("Acceptance", final_accept))
                file.write('%-12s: %d\n' % ("Swaps", swaps))

    def save_info(self, file, epochs, burnin, save_period):
        with open(file, 'w') as f:
            f.write('%-12s: %i\n' % ("# epochs", epochs))
            f.write('%-12s: %i\n' % ("Burnin", burnin))
            f.write('%-12s: %i\n' % ("Save period", save_period))
            f.write('%-12s: %i\n' % ("Dimensions", self.chain[0].D))
            f.write('%-12s: %i\n' % ("# Taxa", self.chain[0].S))
            f.write('%-12s: %i\n' % ("Unique sites", self.chain[0].L))
            f.write('%-12s: %i\n' % ("Chains", self.nChains))
            f.write('%-12s: %s\n' % ("Connect Mthd", self.connect_method))
            for i in range(self.nChains):
                f.write('%-12s: %f\n' % ("Chain temp", self.chain[i].temp))
                f.write('%-12s: %f\n' % ("Step Scale", self.chain[i].step_scale))
            for key, value in self.chain[0].prior.items():
                f.write('%-12s: %f\n' % (key, value))

    def save_iteration(self, path_write, iteration):
        if iteration > 0:
            acceptance = self.chain[0].accepted / self.chain[0].iterations
            print('epoch: %-12i Acceptance Rate: %5.3f' % (iteration, acceptance), end="", flush=True)
            if self.nChains > 1:
                print(" (", end="")
                for c in range(self.nChains-1):
                    print(' %5.3f' % (self.chain[c+1].accepted / self.chain[c+1].iterations), end="")
                print(")")
            else:
                print("")
        tree.save_tree(path_write, 'mcmc', self.chain[0].peel, self.chain[0].blens,
                       iteration*self.chain[0].bcount, self.chain[0].lnP)
        fn = path_write + '/locations.csv'
        with open(fn, 'a') as file:
            file.write(
                np.array2string(self.chain[0].loc_vec.data.numpy())
                .replace('\n', '').replace('[', '').replace(']', '') + "\n")

    def swap(self):
        # randomly swap 2 chains according to MCMCMC
        swappers = torch.multinomial(torch.ones(self.nChains), 2, replacement=False)

        prob1 = (self.chain[swappers[0]].lnP / self.chain[swappers[1]].lnP)**self.chain[swappers[1]].temp
        prob2 = (self.chain[swappers[1]].lnP / self.chain[swappers[0]].lnP)**self.chain[swappers[0]].temp
        alpha = torch.minimum(torch.ones(1), prob1 * prob2)

        if alpha > uniform.Uniform(torch.zeros(1), torch.ones(1)).rsample():
            temp_ = self.chain[swappers[0]]
            self.chain[swappers[0]] = self.chain[swappers[1]]
            self.chain[swappers[1]] = temp_
            del temp_
            return 1
        return 0

    def initialise_chains(self, emm, n_grids=10, n_trials=10, max_scale=2):
        # initialise each chain
        for i in range(self.nChains):
            if self.connect_method == 'mst':
                int_r, int_dir = self.chain[i].initialise_ints(
                    emm, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale)
                r = np.concatenate((emm["r"], int_r))
                directional = np.concatenate((emm["directional"], int_dir))
            elif self.connect_method == 'geodesics' or self.connect_method == 'incentre':
                r = emm["r"]
                directional = emm["directional"]

            # store in tangent plane R^dim
            loc_poin = utils.dir_to_cart(torch.from_numpy(r), torch.from_numpy(directional))
            self.chain[i].loc = p2t0(loc_poin)
            self.chain[i].n_points = len(self.chain[i].loc)

    @staticmethod
    def run(dim, partials, weights, dists, path_write=None,
            epochs=1000, step_scale=0.01, save_period=1, burnin=0,
            n_grids=10, n_trials=100, nChains=1, connect_method='incentre',
            embed_method='wrap', **prior):
        print('\nRunning Dodonaphy MCMC')
        assert connect_method in ['incentre', 'mst', 'geodesics']

        # embed tips with distances using Hydra
        emm_tips = hydra.hydra(dists, dim=dim, equi_adj=0.5, stress=True)
        print('Embedding Stress (tips only) = {:.4}'.format(emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyMCMC(
            partials, weights, dim, step_scale=step_scale, nChains=nChains,
            connect_method=connect_method, embed_method=embed_method, **prior)

        # Choose internal node locations from best random initialisation
        mymod.initialise_chains(emm_tips, n_grids=n_grids, n_trials=n_trials, max_scale=5)

        # Learn
        mymod.learn(epochs, burnin=burnin, path_write=path_write, save_period=save_period)
