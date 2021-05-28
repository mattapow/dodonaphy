from .model import DodonaphyModel
from .utils import utilFunc
from .hyperboloid import t02p
from numpy.random import normal, uniform
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm


class Mcmc(DodonaphyModel):

    def __init__(self, partials, weights, D, loc):
        DodonaphyModel.__init__(self, partials, weights, D)
        self.loc = loc  # in tangent space t_0

    def learn(self, n_steps, burnin=0, path_write='./out', save_period=1, step_scale=0.01):
        self.step_scale = step_scale

        # os.makedirs(path_write, exist_ok=True)
        os.mkdir(path_write)
        self.save_period = save_period

        fn = path_write + '/' + 'mcmc.info'
        with open(fn, 'w') as file:
            file.write('# steps:     ' + str(n_steps) + '\n')
            file.write('Burnin:      ' + str(burnin) + '\n')
            file.write('Save period: ' + str(save_period) + '\n')
            file.write('Step scale:  ' + str(step_scale) + '\n')
            file.write('Dimensions:  ' + str(self.D) + '\n')
            file.write('# Taxa:      ' + str(self.S) + '\n')
            file.write('Seq. length: ' + str(self.L) + '\n')

        for _ in range(burnin):
            self.step()

        accepted = 0
        if self.D == 2:
            _, ax = plt.subplots(1, 1, sharex=True, sharey=True)
            cmap = matplotlib.cm.get_cmap('plasma')
        for i in range(n_steps):
            # step
            accepted += self.step()

            # set peel + blens + poincare locations
            loc_poin = t02p(self.loc, np.zeros_like(self.loc), self.D)
            leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_poin)
            self.peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
            self.blens = super().compute_branch_lengths(self.S, self.D, self.peel, leaf_r, leaf_dir, int_r, int_dir)
            loc_poin = np.concatenate((loc_poin, np.expand_dims(loc_poin[0, :], axis=0)))

            # plot
            if self.S == 2:
                utilFunc.plot_tree(ax, self.peel, loc_poin, color=cmap(i / n_steps), labels=False)

            # save
            if i % self.save_period == 0:
                print('Iteration: ' + str(i) + ' / ' + str(n_steps) + '   Acceptance Rate: ' + str(accepted/(i+1)))
                utilFunc.save_tree(path_write, 'mcmc', self.peel, self.blens)

        print('Acceptance ratio: ' + str(accepted/n_steps))

        if self.D == 2:
            utilFunc.plot_tree(ax, self.peel, loc_poin, color=cmap(i / n_steps), labels=True)
            plt.show()

    def step(self):
        loc_proposal = self.loc + normal(0, self.step_scale, size=(2*self.S - 2, self.D))
        r = self.accept_ratio(loc_proposal)
        # TODO: maybe move each node one at a time?

        accept = False
        if r >= 1:
            accept = True
        elif uniform() < r:
            accept = True
        if accept:
            self.loc = loc_proposal
        return int(accept)

    def accept_ratio(self, loc_proposal):
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(self.loc)
        like_current = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir).detach().numpy()
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_proposal)
        like_proposal = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir).detach().numpy()
        like_ratio = np.exp(like_proposal - like_current)

        # TODO: priors?
        # log(distToOrigin)
        # log(distToReferenceSeq)
        # log(distToFather)
        prior_ratio = 1

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        return np.minimum(1., prior_ratio * like_ratio * hastings_ratio)
