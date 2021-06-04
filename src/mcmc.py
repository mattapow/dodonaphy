from .utils import utilFunc
from .hyperboloid import t02p
from .base_model import BaseModel
from torch.distributions import normal, uniform
import torch


class Mcmc(BaseModel):

    def __init__(self, partials, weights, dim, loc, **prior):
        super().__init__(partials, weights, dim, **prior)
        self.loc = loc

    def learn(self, epochs, burnin=0, path_write='./out', save_period=1, step_scale=0.01):
        # TODO: add a hot chain?
        self.step_scale = step_scale
        self.save_period = save_period

        fn = path_write + '/' + 'mcmc.info'
        with open(fn, 'w') as file:
            file.write('# epochs:     ' + str(epochs) + '\n')
            file.write('Burnin:      ' + str(burnin) + '\n')
            file.write('Save period: ' + str(save_period) + '\n')
            file.write('Step scale:  ' + str(step_scale) + '\n')
            file.write('Dimensions:  ' + str(self.D) + '\n')
            file.write('# Taxa:      ' + str(self.S) + '\n')
            file.write('Seq. length: ' + str(self.L) + '\n')

        fn = path_write + '/mcmc.trees'
        with open(fn, 'w') as file:
            file.write("#NEXUS\n\n")
            file.write("Begin taxa;\n\tDimensions ntax=" + str(self.S) + ";\n")
            file.write("\tTaxlabels\n")
            for i in range(self.S):
                file.write("\t\t" + "T" + str(i+1) + "\n")
            file.write("\t\t;\nEnd;\n\n")
            file.write("Begin trees;\n")

        for _ in range(burnin):
            self.evolove()

        accepted = 0

        for i in range(epochs):
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
                    print('Epoch: %i / %i\tAcceptance Rate: %.3f' % (i, epochs, accepted/i))
                else:
                    self.lnP = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)
                utilFunc.save_tree(path_write, 'mcmc', self.peel, self.blens, i*self.bcount, self.lnP)
                fn = path_write + '/locations.txt'
                with open(fn, 'a') as file:
                    file.write(str(loc_vec.data.numpy()).replace('\n', '').replace('[', '').replace(']', '') + "\n")

            # step
            accepted += self.evolve()

        fn = path_write + '/' + 'mcmc.info'
        with open(fn, 'a') as file:
            file.write('Acceptance:   ' + str(epochs) + '\n')

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
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(self.loc)
        current_like = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)
        current_prior = self.compute_prior(leaf_r, leaf_dir, int_r, int_dir, **self.prior)

        # proposal likelihood + prior
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_proposal)
        prop_like = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir)
        prop_prior = self.compute_prior(leaf_r, leaf_dir, int_r, int_dir, **self.prior)

        # likelihood ratio
        like_ratio = torch.exp(prop_like - current_like)

        # prior ratio
        prior_ratio = prop_prior / current_prior

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        return torch.minimum(torch.ones(1), prior_ratio * like_ratio * hastings_ratio), prop_like
