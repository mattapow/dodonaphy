import torch

from dodonaphy import Chyp_torch, peeler, tree
from dodonaphy.hmap import HMAP


class Laplace(HMAP):

    def get_ln_posterior(self, leaf_loc_flat):
        """Returns the posterior value.

        Assumes phylo likelihood as model.

        Args:
            leaf_loc ([type]): [description]
            curvature ([type]): [description]
        """
        leaf_loc = leaf_loc_flat.view((self.S, self.D))
        dist_2d = Chyp_torch.get_pdm(
            leaf_loc, curvature=self.curvature, matsumoto=self.matsumoto
        )
        peel, blens = peeler.nj_torch(dist_2d, tau=self.soft_temp)
        ln_p = self.compute_LL(peel, blens, self.phylomodel.sub_rates, self.phylomodel.freqs)
        ln_prior = self.compute_prior_gamma_dir(blens)
        return ln_p + ln_prior

    def laplace(self, n_samples=100):
        """Generate a laplace approximation around the current embedding.

        Args:
            n (int, optional): Number of samples. Defaults to 100.
        """
        hessian = torch.autograd.functional.hessian
        normal = torch.distributions.multivariate_normal.MultivariateNormal

        print("Generating laplace approximation: ", end="", flush=True)
        filename = "laplace_samples"
        tree.save_tree_head(self.path_write, filename, self.tip_labels)
        mean = self.get_locs().view(-1)
        for smp_i in range(n_samples):
            res = hessian(self.get_ln_posterior, mean, vectorize=True)
            cov = -torch.linalg.inv(res)
            norm_aprx = normal(mean, cov)
            sample = norm_aprx.sample(torch.Size((1,)))
            dists = Chyp_torch.get_pdm(
                sample, curvature=self.curvature, matsumoto=self.matsumoto
            )
            peel, blens = peeler.nj_torch(dists)
            blens = torch.tensor(blens)
            ln_p = self.compute_LL(peel, blens, self.phylomodel.sub_rates, self.phylomodel.freqs)
            ln_prior = self.compute_prior_gamma_dir(blens)
            if self.path_write is not None:
                tree.save_tree(
                    self.path_write,
                    filename,
                    peel,
                    blens,
                    smp_i,
                    ln_p,
                    ln_prior,
                    self.name_id,
                )
        print("done.")
