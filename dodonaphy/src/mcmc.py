from . import utils, tree, hydra, peeler, hyperboloid
from .base_model import BaseModel

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import uniform
import torch
import numpy as np
import os


class Chain(BaseModel):
    def __init__(self, partials, weights, dim, leaf_r=None, leaf_dir=None, int_r=None, int_dir=None, step_scale=0.01,
                 temp=1, target_acceptance=.234, connect_method='mst', embed_method='simple', **prior):
        super().__init__(partials, weights, dim, **prior)
        self.leaf_dir = leaf_dir  # S x D
        self.int_dir = int_dir  # S-2 x D
        self.int_r = int_r  # S-2
        self.leaf_r = leaf_r  # single scalar
        if leaf_dir is not None:
            self.S = len(leaf_dir)

        assert embed_method in ('simple', 'wrap')
        self.embed_method = embed_method
        assert connect_method in ("incentre", "geodesics", "mst", "nj", "mst_choice")
        self.connect_method = connect_method

        self.step_scale = step_scale
        self.temp = temp
        self.accepted = 0
        self.iterations = 0
        self.target_acceptance = target_acceptance
        self.converged = [False] * 200
        self.moreTune = True

    def set_probability(self):
        # initialise likelihood and prior values of embedding

        # set peel + poincare locations
        if self.connect_method in ('geodesics', 'incentre'):
            loc_poin = self.leaf_dir * self.leaf_r
            self.peel, int_locs = peeler.make_peel_tips(loc_poin, self.connect_method)
            self.int_r, self.int_dir = utils.cart_to_dir(int_locs)
            leaf_r_all, self.leaf_dir = utils.cart_to_dir(loc_poin)
            self.leaf_r = leaf_r_all[0]
        elif self.connect_method == 'nj':
            self.peel, self.blens = peeler.nj(self.leaf_r.repeat(self.S), self.leaf_dir)
        elif self.connect_method == 'mst':
            self.peel = peeler.make_peel_mst(self.leaf_r.repeat(self.S), self.leaf_dir, self.int_r, self.int_dir)
        elif self.connect_method == 'mst_choice':
            self.peel = self.select_peel_mst(self.leaf_r.repeat(self.S), self.leaf_dir, self.int_r, self.int_dir)

        # set blens
        if self.connect_method != 'nj':
            self.blens = self.compute_branch_lengths(
                self.S, self.peel, self.leaf_r.repeat(self.S), self.leaf_dir, self.int_r, self.int_dir)

        # current likelihood
        # self.lnP = self.compute_log_a_like(self.leaf_r.repeat(self.S), self.leaf_dir)
        self.lnP = self.compute_LL(self.peel, self.blens)

        # current prior
        # self.lnPrior = self.compute_prior_birthdeath(self.peel, self.blens, **self.prior)
        self.lnPrior = self.compute_prior_gamma_dir(self.blens)

    def evolve(self):
        # propose new embedding
        proposal = self.propose()

        # Decide whether to accept proposal
        r = self.accept_ratio(proposal)

        accept = False
        if r >= 1:
            accept = True
        elif uniform.Uniform(torch.zeros(1), torch.ones(1)).sample() < r:
            accept = True

        if accept:
            self.leaf_r = proposal['leaf_r']
            self.leaf_dir = proposal['leaf_dir']
            self.lnP = proposal['lnP']
            self.lnPrior = proposal['lnPrior']
            self.peel = proposal['peel']
            self.blens = proposal['blens']
            if self.connect_method == 'mst':
                self.int_r = proposal['int_r']
                self.int_dir = proposal['int_dir']
            self.accepted += 1
        self.iterations += 1

        return accept

    def propose(self):
        # TODO: propose unit directionals on sphere
        n_leaf_vars = self.S * self.D
        sample = MultivariateNormal(
            torch.zeros(n_leaf_vars, dtype=torch.double),
            torch.eye(n_leaf_vars, dtype=torch.double) * self.step_scale).sample().reshape(self.S, self.D)
        if self.embed_method == 'simple':
            # transform leaves to R^n
            leaf_loc_t0 = utils.ball2real(self.leaf_r * self.leaf_dir).clone()

            # propose new leaf nodes from normal in R^n
            leaf_loc_t0 = leaf_loc_t0 + sample

            # normalise leaves to sphere with radius leaf_r_prop = first leaf radii
            leaf_r_t0 = torch.norm(leaf_loc_t0[0, :])
            log_abs_det_jacobian = utils.normalise_LADJ(leaf_loc_t0) + torch.log(leaf_r_t0)
            leaf_loc_t0 = utils.normalise(leaf_loc_t0) * leaf_r_t0

            # Convert to Ball
            leaf_loc_prop = utils.real2ball(leaf_loc_t0)
            log_abs_det_jacobian = utils.real2ball_LADJ(leaf_loc_t0)

        elif self.embed_method == 'wrap':
            # transform leaves to R^n
            leaf_loc_t0 = hyperboloid.p2t0(self.leaf_r * self.leaf_dir)
            leaf_loc_t0 = leaf_loc_t0.clone()

            # propose new leaf nodes from normal in R^n and convert to poincare ball
            leaf_loc_prop, log_abs_det_jacobian = hyperboloid.t02p(sample, leaf_loc_t0, get_jacobian=True)
        # get r and directional
        leaf_r_prop = torch.norm(leaf_loc_prop[0, :])
        leaf_dir_prop = leaf_loc_prop / torch.norm(leaf_loc_prop, dim=-1, keepdim=True)

        # internal nodes for mst
        if self.connect_method in ('mst', 'mst_choice'):
            n_int_vars = (self.S - 2) * self.D
            sample = MultivariateNormal(
                torch.zeros(n_int_vars, dtype=torch.double),
                torch.eye(n_int_vars, dtype=torch.double) * self.step_scale).sample().reshape(self.S-2, self.D)

            int_loc = self.int_r.reshape(self.S-2, 1).repeat(1, self.D) * self.int_dir
            if self.embed_method == 'simple':
                # transform ints to R^n
                int_loc_t0 = utils.ball2real(int_loc).clone()

                # propose new int nodes from normal in R^n
                int_loc_t0 = int_loc_t0 + sample

                # convert ints to poincare ball
                int_loc_prop = utils.real2ball(int_loc_t0)
                log_abs_det_jacobian = log_abs_det_jacobian + utils.real2ball_LADJ(int_loc_t0)

            elif self.embed_method == 'wrap':
                # transform ints to R^n
                int_loc_t0 = hyperboloid.p2t0(int_loc).clone()

                # propose new int nodes from normal in R^n
                int_loc_prop, int_jacobian = hyperboloid.t02p(sample, int_loc_t0, get_jacobian=True)
                log_abs_det_jacobian = log_abs_det_jacobian + int_jacobian

            # get r and directional
            int_r_prop = torch.norm(int_loc_prop, dim=-1)
            int_dir_prop = int_loc_prop / torch.norm(int_loc_prop, dim=-1, keepdim=True)

            # restrict int_r to less than leaf_r
            # TODO: does this change the hastings ratio?
            int_r_prop_big = int_r_prop[int_r_prop > leaf_r_prop]
            log_abs_det_jacobian = log_abs_det_jacobian + torch.log(leaf_r_prop / int_r_prop_big)
            int_r_prop[int_r_prop > leaf_r_prop] = leaf_r_prop

        # proposal peel and blens
        leaf_r = leaf_r_prop.repeat(self.S)
        if self.connect_method in ('geodesics', 'incentre'):
            peel, int_locs = peeler.make_peel_tips(leaf_r_prop * leaf_dir_prop, connect_method=self.connect_method)
            int_r_prop, int_dir_prop = utils.cart_to_dir(int_locs)
        elif self.connect_method == 'nj':
            peel, blens = peeler.nj(leaf_r, leaf_dir_prop)
        elif self.connect_method == 'mst':
            peel = peeler.make_peel_mst(leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)
        elif self.connect_method == 'mst_choice':
            peel = self.select_peel_mst(leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)

        # get proposal branch lengths
        if self.connect_method != 'nj':
            blens = self.compute_branch_lengths(self.S, peel, leaf_r, leaf_dir_prop, int_r_prop, int_dir_prop)

        if self.connect_method in ('geodesics', 'incentre', 'nj'):
            proposal = {
                'leaf_r': leaf_r_prop,
                'leaf_dir': leaf_dir_prop,
                'peel': peel,
                'blens': blens,
                'jacobian': log_abs_det_jacobian
            }
        else:
            proposal = {
                'leaf_r': leaf_r_prop,
                'leaf_dir': leaf_dir_prop,
                'int_r': int_r_prop,
                'int_dir': int_dir_prop,
                'peel': peel,
                'blens': blens,
                'jacobian': log_abs_det_jacobian
            }
        return proposal

    def accept_ratio(self, p):
        """Acceptance critereon for Metropolis-Hastings

        Args:
            p ([type]): Proposal dictionary

        Returns:
            tuple: (r, prop_like)
            The acceptance ratio r and the likelihood of the proposal.
        """

        p['lnP'] = self.compute_LL(p['peel'], p['blens'])
        # p['lnP'] = self.compute_log_a_like(p['leaf_r'].repeat(self.S), p['leaf_dir'])

        # p['lnPrior'] = self.compute_prior_birthdeath(p['peel'], p['blens'], **self.prior)
        p['lnPrior'] = self.compute_prior_gamma_dir(p['blens'])

        # likelihood ratio
        like_ratio = p['lnP'] - self.lnP

        # prior ratio
        prior_ratio = p['lnPrior'] - self.lnPrior

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        # acceptance ratio
        r = torch.minimum(torch.ones(1),
                          torch.exp((prior_ratio + like_ratio)*self.temp + hastings_ratio))

        return r

    def tune_step(self, tol=0.01):
        """Tune the acceptance rate. Simple Euler method.

        Args:
            tol (float, optional): Tolerance. Defaults to 0.01.
        """
        if not self.moreTune or self.iterations == 0:
            return

        lr = 0.001
        eps = 0.00000001
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

    def __init__(self, partials, weights, dim, connect_method='mst', embed_method='simple',
                 step_scale=0.01, nChains=1, **prior):
        self.nChains = nChains
        self.chain = []
        dTemp = 0.1
        for i in range(nChains):
            temp = 1./(1+dTemp*i)
            self.chain.append(
                Chain(partials, weights, dim, step_scale=step_scale, temp=temp, embed_method=embed_method,
                      connect_method=connect_method, **prior))

    def learn(self, epochs, burnin=0, path_write='./out', save_period=1):
        print("Using 1 cold chain and %d hot chains." % int(self.nChains-1))
        self.save_period = save_period

        if path_write is not None:
            info_file = path_write + '/' + 'mcmc.info'
            self.save_info(info_file, epochs, burnin, save_period)
            tree.save_tree_head(path_write, "mcmc", self.chain[0].S)

        # Initialise prior and likelihood
        for c in range(self.nChains):
            self.chain[c].set_probability()

        if burnin > 0:
            print('Burning in for %d iterations.' % burnin)
            deceile = 1
            for i in range(burnin):
                if i / burnin * 10 > deceile:
                    print("%d%% " % (deceile*10), end="", flush=True)
                    deceile += 1
                for c in range(self.nChains):
                    # step
                    self.chain[c].evolve()
                    # tune step
                    self.chain[c].tune_step()

                # swap 2 chains
                if self.nChains > 1:
                    _ = self.swap()
            print("100%")

        swaps = 0
        print("Running for %d epochs.\n" % epochs)
        for i in range(epochs):
            for c in range(self.nChains):
                # step
                self.chain[c].evolve()
                # tune step if not converged
                self.chain[c].tune_step()

            # save
            doSave = self.save_period > 0 and i % self.save_period == 0
            if doSave:
                if path_write is not None:
                    self.save_iteration(path_write, i)

                if i > 0:
                    print('epoch: %-12i LnL: %10.1f Acceptance Rate: %5.3f' %
                          (i, self.chain[0].lnP, self.chain[0].accepted / self.chain[0].iterations),
                          end="", flush=True)

                    if self.nChains > 1:
                        print(" (", end="")
                        for c in range(self.nChains-1):
                            print(' %5.3f' % (self.chain[c+1].accepted / self.chain[c+1].iterations), end="")
                        print(")")
                    else:
                        print("")

            # swap 2 chains
            if self.nChains > 1:
                swaps += self.swap()

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
            for i in range(self.nChains):
                f.write('%-12s: %f\n' % ("Chain temp", self.chain[i].temp))
                f.write('%-12s: %f\n' % ("Step Scale", self.chain[i].step_scale))
                f.write('%-12s: %s\n' % ("Connect Mthd", self.chain[i].connect_method))
                f.write('%-12s: %s\n' % ("Embed Mthd", self.chain[i].embed_method))
            for key, value in self.chain[0].prior.items():
                f.write('%-12s: %f\n' % (key, value))

    def save_iteration(self, path_write, iteration):
        lnP = self.chain[0].compute_LL(self.chain[0].peel, self.chain[0].blens)
        tree.save_tree(path_write, 'mcmc', self.chain[0].peel, self.chain[0].blens,
                       iteration, float(lnP), float(self.chain[0].lnPrior))
        fn = path_write + '/locations.csv'
        if not os.path.isfile(fn):
            with open(fn, 'a') as file:
                file.write("leaf_r, ")
                for i in range(len(self.chain[0].leaf_dir)):
                    for j in range(self.chain[0].D):
                        file.write("leaf_%d_dir_%d, " % (i, j))
                if self.chain[0].int_r is not None:
                    for i in range(len(self.chain[0].int_r)):
                        file.write('int_%d_r, ' % (i))
                    for i in range(len(self.chain[0].int_dir)):
                        for j in range(self.chain[0].D):
                            file.write('int_%d_dir_%d' % (i, j))
                            if not (j == self.chain[0].D - 1 and i == len(self.chain[0].int_dir)):
                                file.write(', ')
                file.write("\n")

        with open(fn, 'a') as file:
            file.write(np.array2string(self.chain[0].leaf_r.data.numpy(), separator=', ')
                       .replace('\n', '').replace('[', '').replace(']', ''))
            file.write(', ')
            file.write(np.array2string(self.chain[0].leaf_dir.data.numpy(), separator=', ')
                       .replace('\n', '').replace('[', '').replace(']', ''))

            if self.chain[0].connect_method == 'mst':
                file.write(', ')
                file.write(np.array2string(self.chain[0].int_r.data.numpy(), separator=', ')
                           .replace('\n', '').replace('[', '').replace(']', ''))
                file.write(',')
                file.write(np.array2string(self.chain[0].int_dir.data.numpy(), separator=', ')
                           .replace('\n', '').replace('[', '').replace(']', ''))
            file.write("\n")

    def swap(self):
        """
        randomly swap states in 2 chains according to MCMCMC
        """

        # Pick two adjacent chains
        i = torch.multinomial(torch.ones(self.nChains-1), 1, replacement=False)
        j = i + 1

        # get log posterior (unnormalised)
        lnPost_i = self.chain[i].lnP + self.chain[i].lnPrior
        lnPost_j = self.chain[j].lnP + self.chain[j].lnPrior

        # probability of exhanging these two chains
        prob1 = (lnPost_i - lnPost_j) * self.chain[j].temp
        prob2 = (lnPost_j - lnPost_i) * self.chain[i].temp
        r = torch.minimum(torch.ones(1), torch.exp(prob1 + prob2))

        # swap with probability r
        if r > uniform.Uniform(torch.zeros(1), torch.ones(1)).rsample():
            # swap the locations and current probability
            self.chain[i].leaf_r, self.chain[j].leaf_r = self.chain[j].leaf_r, self.chain[i].leaf_r
            self.chain[i].leaf_dir, self.chain[j].leaf_dir = self.chain[j].leaf_dir, self.chain[i].leaf_dir
            self.chain[i].int_r, self.chain[j].int_r = self.chain[j].int_r, self.chain[i].int_r
            self.chain[i].int_dir, self.chain[j].int_dir = self.chain[j].int_dir, self.chain[i].int_dir
            self.chain[i].lnP, self.chain[j].lnP = self.chain[j].lnP, self.chain[i].lnP
            self.chain[i].lnPrior, self.chain[j].lnPrior = self.chain[j].lnPrior, self.chain[i].lnPrior
            return 1
        return 0

    def initialise_chains(self, emm, n_grids=10, n_trials=10, max_scale=2):
        # initialise each chain
        for i in range(self.nChains):
            # put leaves on a sphere
            self.chain[i].leaf_r = torch.tensor(np.mean(emm["r"], dtype=np.double))
            self.chain[i].leaf_dir = torch.from_numpy(emm["directional"].astype(np.double))
            self.chain[i].n_points = len(self.chain[i].leaf_dir)
            self.chain[i].int_r = None
            self.chain[i].int_dir = None

            if self.chain[i].connect_method in ('mst', 'mst_choice'):
                int_r, int_dir = self.chain[i].initialise_ints(
                    emm, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale)
                self.chain[i].int_r = torch.from_numpy(int_r.astype(np.double))
                self.chain[i].int_dir = torch.from_numpy(int_dir.astype(np.double))

    @staticmethod
    def run(dim, partials, weights, dists, path_write=None,
            epochs=1000, step_scale=0.01, save_period=1, burnin=0,
            n_grids=10, n_trials=100, max_scale=1, nChains=1,
            connect_method='mst', embed_method='simple', **prior):
        print('\nRunning Dodonaphy MCMC')
        assert connect_method in ['incentre', 'mst', 'geodesics', 'nj', 'mst_choice']

        # embed tips with distances using Hydra
        emm_tips = hydra.hydra(dists, dim=dim, stress=True, **{'isotropic_adj': True})
        print('Embedding Stress (tips only) = {:.4}'.format(emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyMCMC(
            partials, weights, dim, step_scale=step_scale, nChains=nChains,
            connect_method=connect_method, embed_method=embed_method, **prior)

        # Choose internal node locations from best random initialisation
        mymod.initialise_chains(emm_tips, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale)

        # Learn
        mymod.learn(epochs, burnin=burnin, path_write=path_write, save_period=save_period)
