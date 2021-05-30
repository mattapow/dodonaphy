from .utils import utilFunc
from .hyperboloid import t02p
from .phylo import calculate_treelikelihood, JC69_p_t
from torch.distributions import normal, uniform
import torch
import matplotlib.pyplot as plt
import matplotlib.cm


class Mcmc(object):

    def __init__(self, partials, weights, D, loc):
        self.partials = partials
        self.weights = weights
        self.D = D
        self.loc = loc  # in tangent space t_0

        self.S = len(partials)
        self.L = partials[0].shape[1]
        self.bcount = 2 * self.S - 2

    def learn(self, n_steps, burnin=0, path_write='./out', save_period=1, step_scale=0.01):
        self.step_scale = step_scale
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

        fn = path_write + '/mcmc.trees'
        with open(fn, 'w') as file:
            file.write("#NEXUS\n")
            file.write("Begin taxa;\tDimensions ntax=" + str(self.S) + ";\n")
            file.write("\t\tTaxlabels\n")
            for i in range(self.S):
                file.write("\t\t" + "T" + str(i+1) + "\n")
            file.write("End;")

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
            loc_poin = t02p(self.loc, torch.zeros_like(self.loc), self.D)
            leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_poin)
            self.peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)
            self.blens = super().compute_branch_lengths(self.S, self.D, self.peel, leaf_r, leaf_dir, int_r, int_dir)
            loc_poin = torch.cat((loc_poin, torch.unsqueeze(loc_poin[0, :], axis=0)))

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
        loc_proposal = self.loc + normal.Normal(0, self.step_scale, size=(2*self.S - 2, self.D))
        r = self.accept_ratio(loc_proposal)
        # TODO: maybe move each node one at a time?

        accept = False
        if r >= 1:
            accept = True
        elif uniform.Uniform(torch.zeros(1), torch.ones(1)) < r:
            accept = True
        if accept:
            self.loc = loc_proposal
        return int(accept)

    def accept_ratio(self, loc_proposal):
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(self.loc)
        like_current = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir).detach().numpy()
        leaf_r, int_r, leaf_dir, int_dir = utilFunc.cart_to_dir_tree(loc_proposal)
        like_proposal = self.compute_LL(leaf_r, leaf_dir, int_r, int_dir).detach().numpy()
        like_ratio = torch.exp(like_proposal - like_current)

        # TODO: priors?
        # log(distToOrigin)
        # log(distToReferenceSeq)
        # log(distToFather)
        prior_ratio = 1

        # Proposals are symmetric Guassians
        hastings_ratio = 1

        return torch.minimum(1., prior_ratio * like_ratio * hastings_ratio)

    def compute_branch_lengths(self, S, dim, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=1):
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
            directional2 = int_dir[peel[b][2]-S-1, ]
            r2 = int_r[peel[b][2]-S-1]

            for i in range(2):
                if peel[b][i] < S:
                    # leaf to internal
                    r1 = leaf_r[peel[b][i]]
                    directional1 = leaf_dir[peel[b][i], :]
                else:
                    # internal to internal
                    r1 = int_r[peel[b][i]-S-1]
                    directional1 = int_dir[peel[b][i]-S-1, ]

                hd = utilFunc.hyperbolic_distance(
                    r1, r2, directional1, directional2, curvature)

                # apply the inverse transform from Matsumoto et al 2020
                hd = torch.log(torch.cosh(hd))

                # add a tiny amount to avoid zero-length branches
                eps = torch.finfo(torch.double).eps
                blens[peel[b][i]] = torch.maximum(hd, eps)

        return blens

    def compute_LL(self, leaf_r, leaf_dir, int_r, int_dir):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """

        peel = utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir)

        # branch lengths
        blens = self.compute_branch_lengths(self.S, self.D, peel, leaf_r, leaf_dir, int_r, int_dir)

        mats = JC69_p_t(blens)
        return calculate_treelikelihood(
            self.partials, self.weights, peel, mats, torch.full([4], 0.25, dtype=torch.float64))
