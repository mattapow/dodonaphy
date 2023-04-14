import os
import time
import warnings
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from dodonaphy import Chyp_torch, peeler, tree, utils
from dodonaphy.base_model import BaseModel
from dodonaphy.phylo import calculate_treelikelihood
from hydraPlus import hydraPlus


class DodonaphyVI(BaseModel):
    def __init__(
        self,
        partials,
        weights,
        dim,
        embedder="wrap",
        connector="nj",
        curvature=-1.0,
        soft_temp=None,
        noise=None,
        truncate=None,
        tip_labels=None,
        n_boosts=1,
        start="",
        model_name="JC69",
    ):
        super().__init__(
            "vi",
            partials,
            weights,
            dim,
            soft_temp=soft_temp,
            embedder=embedder,
            connector=connector,
            curvature=curvature,
            tip_labels=tip_labels,
            model_name=model_name,
        )
        print("Initialising variational model.\n")

        # Store distribution centres mu in hyperboloid projected onto R^dim.
        # The last coordinate in R^(dim+1) is determined.
        self.n_boosts = n_boosts

        self.noise = noise
        self.truncate = truncate
        self.start = start
        # Variational parameters must be set using set_variationalParams() or set_variationalParams_random()
        self.VariationalParams = dict()

    def set_variationalParams_random(self):
        mix_weights = np.full((self.n_boosts), 1 / self.n_boosts)
        leaf_sigma = np.random.exponential(size=(self.n_boosts, self.S, self.D))
        int_sigma = np.random.exponential(size=(self.n_boosts, self.S - 2, self.D))

        self.VariationalParams = {
            "leaf_mu": torch.randn(
                (self.n_boosts, self.S, self.D),
                requires_grad=True,
                dtype=torch.float64,
            ),
            "leaf_sigma": torch.tensor(
                leaf_sigma, requires_grad=True, dtype=torch.float64
            ),
            "mix_weights": torch.tensor(
                mix_weights, requires_grad=True, dtype=torch.float64
            ),
        }
        if self.internals_exist:
            self.VariationalParams["int_mu"] = torch.randn(
                (self.n_boosts, self.S - 2, self.D),
                requires_grad=True,
                dtype=torch.float64,
            )
            self.VariationalParams["int_sigma"] = torch.tensor(
                int_sigma, requires_grad=True, dtype=torch.float64
            )
        # set evolutionary model parameters to optimise
        if not self.phylomodel.fix_sub_rates:
            self.VariationalParams["sub_rates"] = self.phylomodel.sub_rates
        if not self.phylomodel.fix_freqs:
            self.VariationalParams["freqs"] = self.phylomodel.freqs

    def set_variationalParams(self, param_init):
        # set dimensions of input
        if param_init["leaf_mu"].ndim == 2:
            param_init["leaf_mu"].unsqueeze(0)
        if param_init["leaf_sigma"].ndim == 2:
            param_init["leaf_sigma"].unsqueeze(0)
        # set leaf mean locations
        self.VariationalParams["leaf_mu"] = (
            param_init["leaf_mu"].repeat((self.n_boosts, 1, 1)).requires_grad_()
        )
        # set leaf scale (sigma in normal distribution)
        self.VariationalParams["leaf_sigma"] = (
            param_init["leaf_sigma"].repeat((self.n_boosts, 1, 1)).requires_grad_()
        )
        if self.internals_exist:
            if param_init["int_mu"].ndim == 2:
                param_init["int_mu"].unsqueeze(0)
            if param_init["int_sigma"].ndim == 2:
                param_init["int_sigma"].unsqueeze(0)
            self.VariationalParams["int_mu"] = (
                param_init["int_mu"].repeat((self.n_boosts, 1, 1)).requires_grad_()
            )
            self.VariationalParams["int_sigma"] = (
                param_init["int_sigma"].repeat((self.n_boosts, 1, 1)).requires_grad_()
            )

        if "mix_weights" in param_init.keys():
            self.VariationalParams["mix_weights"] = param_init[
                "mix_weights"
            ].requires_grad_()
        else:
            # default to 1 mixture
            self.VariationalParams["mix_weights"] = torch.tensor(
                np.ones((1)), dtype=torch.float64
            ).requires_grad_()

        # set evolutionary model parameters to optimise
        if not self.phylomodel.fix_sub_rates:
            self.VariationalParams["sub_rates"] = self.phylomodel.sub_rates
        if not self.phylomodel.fix_freqs:
            self.VariationalParams["freqs"] = self.phylomodel.freqs

    def draw_sample(self, nSample=100, **kwargs):
        """Draw samples from the variational posterior distribution

        Args:
            nSample (int, optional): Number of samples to be drawn. Defaults to 100.

        Returns:
            tuple[list list list list]: peel, blens, location, lp. If kwarg 'lp' is passed.
            Locations are in Hyperbolic space.
            tuple[list list list]: peel, blens, location, lp. Otherwise.
        """
        for key in kwargs.keys():
            assert key in ("get_likelihood", "get_elbo")

        # make peel, blens and X for each of these samples
        peel = []
        blens = []
        location = []
        log_like = []
        log_jac = []
        log_prior = []
        log_Q = []

        weights = torch.softmax(self.VariationalParams["mix_weights"], dim=0)
        mix_samples = torch.multinomial(weights, num_samples=nSample, replacement=True)
        for i in range(nSample):
            mix_idx = mix_samples[i]
            n_tip_params = torch.numel(self.VariationalParams["leaf_mu"][mix_idx])
            leaf_loc = self.VariationalParams["leaf_mu"][mix_idx].reshape(n_tip_params)
            leaf_cov = torch.eye(
                n_tip_params, dtype=torch.double
            ) * self.VariationalParams["leaf_sigma"][mix_idx].exp().reshape(
                n_tip_params
            )
            if self.internals_exist:
                n_int_params = torch.numel(self.VariationalParams["int_mu"])
                int_loc = self.VariationalParams["int_mu"][mix_idx].reshape(
                    n_int_params
                )
                int_cov = torch.eye(
                    n_int_params, dtype=torch.double
                ) * self.VariationalParams["int_sigma"][mix_idx].exp().reshape(
                    n_int_params
                )
                sample = self.rsample_tree(leaf_loc, leaf_cov, int_loc, int_cov)
            else:
                sample = self.rsample_tree(leaf_loc, leaf_cov)

            peel.append(sample["peel"])
            blens.append(sample["blens"])
            location.append(sample["leaf_locs"])
            if self.internals_exist:
                location.append(sample["int_locs"])

            if kwargs.get("get_likelihood") or kwargs.get("get_elbo"):
                # regardless of the loss function, compute the likelihood under the phylogenetic model of evolution
                mats = self.phylomodel.get_transition_mats(
                    sample["blens"], self.phylomodel.sub_rates, self.phylomodel.freqs
                )
                freqs = torch.full([4], 0.25, dtype=torch.float64)
                LL = calculate_treelikelihood(
                    self.partials,
                    self.weights,
                    sample["peel"],
                    mats,
                    freqs,
                )
                log_like.append(LL)
                log_prior.append(sample["ln_prior"])
            if kwargs.get("get_elbo"):
                log_Q.append(sample["logQ"])
                log_jac.append(sample["jacobian"])

        if kwargs.get("get_elbo"):
            return peel, blens, location, log_like, log_prior, log_Q, log_jac
        elif kwargs.get("get_likelihood"):
            return peel, blens, location, log_like, log_prior
        else:
            return peel, blens, location

    def calculate_elbo(self, mix_idx):
        """Calculate the elbo of a sample from the variational distributions q_k

        Args:
            mix_idx (int): index of mixture

        Returns:
            float: The evidence lower bound of a sample from q
        """
        n_tip_params = torch.numel(self.VariationalParams["leaf_mu"][mix_idx])
        leaf_locs = self.VariationalParams["leaf_mu"][mix_idx].reshape(n_tip_params)
        leaf_cov = torch.eye(n_tip_params, dtype=torch.double) * self.VariationalParams[
            "leaf_sigma"
        ][mix_idx].exp().reshape(n_tip_params)
        if self.internals_exist:
            n_int_params = torch.numel(self.VariationalParams["int_mu"][mix_idx])
            int_locs = self.VariationalParams["int_mu"][mix_idx].reshape(n_int_params)
            int_cov = torch.eye(
                n_int_params, dtype=torch.double
            ) * self.VariationalParams["int_sigma"][mix_idx].exp().reshape(n_int_params)
            sample = self.rsample_tree(leaf_locs, leaf_cov, int_locs, int_cov)
        else:
            sample = self.rsample_tree(leaf_locs, leaf_cov)

        if sample["jacobian"] == -torch.inf:
            warnings.warn("Jacobian determinant set to zero.")
            sample["jacobian"] = 0.0
        return sample["ln_p"] + sample["ln_prior"] - sample["logQ"] + sample["jacobian"]

    def learn(
        self,
        epochs=1000,
        importance_samples=1,
        path_write="./out",
        lr=1e-3,
        n_draws=100,
    ):
        """Learn the variational parameters using Adam optimiser
        Args:
            epochs (int, optional): Number of epochs. Defaults to 1000.
            importance_samples (int, optional): Number of tree samples at each
            path_write (str): to save output
            lr (float): learning rate for Adam
            n_draws(int): number of samples from final variational distribution
        """
        print(f"Using {importance_samples} tree samples at each epoch.")
        print(f"Using {self.n_boosts} variational distributions for boosting.")
        print(f"Running for {epochs} epochs.\n")
        start_time = time.time()

        # TODO: should probably put this in the super method since super().log() depends on it
        self.path_write = path_write
        if path_write is not None:
            self.log_run_start(path_write, epochs, importance_samples, lr)
            self.elbo_fn = os.path.join(path_write, "elbo.txt")

        def lr_lambda(epoch):
            return 1.0 / np.sqrt(epoch + 1)

        # Consider using LBFGS, but appears to not perform as well.
        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = -self.elbo_siwae(importance_samples)
            if loss.requires_grad:
                loss.backward()
            elbo_hist.append(-loss.detach().item())
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            print("epoch %-12i ELBO: %10.3f" % (epoch + 1, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

            if path_write is not None:
                self.log_elbo(elbo_hist[-1])
                fn = os.path.join(path_write, "vi_params", "latest.csv")
                self.save(fn)
                fn = os.path.join(path_write, "vi_params", f"iteration_{epoch+1}.txt")
                with open(fn, "w", encoding="UTF-8") as file:
                    file.write(f"Epoch: {epoch} / {epochs}")

        if epochs > 0 and path_write is not None:
            self.trace(epochs, path_write, elbo_hist)

        if path_write is not None:
            self.compute_final_elbo(path_write, n_draws)
            self.save_final_info(path_write, time.time() - start_time)

    def compute_final_elbo(self, path_write, n_draws):
        # draw samples (one-by-one) from the final distribution and save them
        tree.save_tree_head(path_write, "samples", self.tip_labels)
        log_elbos = torch.zeros((n_draws, self.n_boosts))
        with torch.no_grad():
            (peel, blens, _, log_like, log_prior, log_Q, log_jac) = self.draw_sample(
                n_draws, get_elbo=True
            )
            for i in range(n_draws):
                tree.save_tree(
                    path_write,
                    "samples",
                    peel[i],
                    blens[i],
                    i,
                    log_like[i].item(),
                    log_prior.item(),
                    self.name_id,
                )
                log_elbos[i] = log_like + log_prior - log_Q + log_jac
        # save the final siwae of these samples
        final_elbo = self.elbo_siwae(importance=n_draws, ln_elbos=log_elbos).item()
        self.log("%-12s: %i\n" % ("Final ELBO (100 samples)", final_elbo))
        print("Final ELBO: {}".format(final_elbo))

    def log_run_start(self, path_write, epochs, importance_samples, lr):
        fn = path_write + "/" + "vi.log"
        self.log("%-12s: %i\n" % ("# epochs", epochs))
        self.log("%-12s: %s\n" % ("prior", "Gamma Dirichlet"))
        self.log("%-12s: %i\n" % ("Importance", importance_samples))
        self.log("%-12s: %i\n" % ("# mixtures", self.n_boosts))
        self.log("%-12s: %i\n" % ("Curvature", self.curvature))
        self.log("%-12s: %i\n" % ("Matsumoto", self.matsumoto))
        self.log("%-12s: %f\n" % ("Soft temp", self.soft_temp))
        self.log("%-12s: %f\n" % ("Log10 Soft temp", np.log10(self.soft_temp)))
        self.log("%s: %i\n" % ("Normalise Leaf", self.normalise_leaf))
        self.log("%-12s: %i\n" % ("Dimensions", self.D))
        self.log("%-12s: %i\n" % ("# Taxa", self.S))
        self.log("%-12s: %i\n" % ("Patterns", self.L))
        self.log("%-12s: %f\n" % ("Learn Rate", lr))
        self.log("%-12s: %s\n" % ("Embed Mthd", self.embedder))
        self.log("%-12s: %s\n" % ("Connect Mthd", self.connector))
        self.log("%-12s: %s\n" % ("Start Tree", self.start))

        vi_path = os.path.join(path_write, "vi_params")
        os.mkdir(vi_path)
        fn = os.path.join(vi_path, "start.csv")
        self.save(fn)

    def log_elbo(self, elbo_value):
        with open(self.elbo_fn, "a", encoding="UTF-8") as f:
            f.write("%f\n" % elbo_value)

    def elbo_siwae(self, importance=1, ln_elbos=None):
        """Compute the ELBO.

        Args:
            importance (int, optional): Number of importance samples of elbo (IWAE).
            Defaults to 1.
            ln_elbos (tensor, optional): Provide the precomputed log elbo values. Defaults to None

        Returns:
            [torch.Tensor]: ELBO value
        """
        if ln_elbos is None:
            ln_elbos = torch.zeros((importance, self.n_boosts))
            for k in range(self.n_boosts):
                for t in range(importance):
                    ln_elbos[t, k] = self.calculate_elbo(k)
        loss = torch.logsumexp(
            -torch.log(torch.tensor(importance))
            + torch.log_softmax(self.VariationalParams["mix_weights"], dim=0)
            + ln_elbos,
            dim=1,
        )
        return torch.mean(loss)

    def rsample_tree(
        self,
        leaf_locs,
        leaf_cov,
        normalise_leaf=False,
    ):
        """Sample a nearby tree embedding.

        Each point is transformed R^n (using the self.embedding method), then
        a normal is sampled and transformed back to H^n. A tree is formed using
        the self.connect method.

        A dictionary is  returned containing information about this sampled tree.
        """
        if self.internals_exist:
            # int_cov = torch.eye((self.S - 2) * self.D, dtype=torch.double) * int_cov
            raise DeprecationWarning("Only embed tips.")

        # reshape covariance if single number
        if torch.numel(leaf_cov) == 1:
            leaf_cov = torch.eye(self.S * self.D, dtype=torch.double) * leaf_cov

        # sample Euclidean location
        leaf_locs, log_Q = self.rsample_Euclid(
            leaf_locs, leaf_cov, is_internal=False, normalise_loc=normalise_leaf
        )
        # transform into tree
        peel, blens, pdm = self.connect(leaf_locs)

        # get jacobian
        def get_blens(locs_t0):
            locs_t0_2d = locs_t0.reshape((self.S, self.D))
            _, blens, _ = self.connect(locs_t0_2d)
            return blens

        # TODO: use analytical form
        jacobian = torch.autograd.functional.jacobian(get_blens, (leaf_locs.flatten()))
        log_abs_det_jacobian = torch.log(
            torch.sqrt(torch.abs(torch.linalg.det(jacobian @ jacobian.T)))
        )

        # get loss
        ln_p, ln_prior = self.get_loss(peel, blens, pdm, leaf_locs)

        sample = {
            "leaf_locs": leaf_locs,
            "peel": peel,
            "blens": blens,
            "jacobian": log_abs_det_jacobian,
            "logQ": log_Q,
            "ln_p": ln_p,
            "ln_prior": ln_prior,
        }
        return sample

    def get_loss(self, peel, blens, pdm, leaf_locs):
        if self.loss_fn == "likelihood":
            ln_p = self.compute_LL(
                peel, blens, self.phylomodel.sub_rates, self.phylomodel.freqs
            )
        elif self.loss_fn == "pair_likelihood":
            ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC":
            ln_p = self.compute_hypHC(pdm, leaf_locs)

        ln_prior = self.compute_prior_gamma_dir(blens)
        return ln_p, ln_prior

    def rsample_Euclid(self, locs, cov, is_internal=False, normalise_loc=False):
        """Sample points in Multi-dimensional Euclidean space

        Args:
            locs (tensor): Mean
            cov (tensor): Covariance
            is_internal (bool): If true, samples n_locs - 2 points.

        Returns:
            tuple: locations, log probability
        """
        if is_internal:
            n_locs = self.S - 2
        else:
            n_locs = self.S
        n_vars = n_locs * self.D
        normal_dist = MultivariateNormal(locs.reshape(n_vars).squeeze(), cov)
        sample = normal_dist.rsample()
        loc_t0 = sample.reshape((n_locs, self.D))

        if normalise_loc:
            r_prop = torch.norm(loc_t0[0, :]).repeat(self.S)
            loc_t0 = utils.normalise(loc_t0) * r_prop.repeat((self.D, 1)).T

        log_Q = normal_dist.log_prob(loc_t0.flatten())
        return loc_t0, log_Q

    def connect(self, locs_t0):
        """Transform embedding locations into a NJ tree.

        Args:
            locs_t0 (tensor): Embedding locations in the tanget space

        Returns:
            tensor: post order tranversal, branch lengths
        """
        pdm = Chyp_torch.get_pdm(locs_t0, curvature=self.curvature)
        if self.connector == "geodesics":
            peel, int_locs, blens = peeler.make_soft_peel_tips(
                locs_t0, connector="geodesics", curvature=self.curvature
            )
        elif self.connector == "nj":
            peel, blens = peeler.nj_torch(pdm, tau=self.soft_temp)
        return peel, blens, pdm

    @staticmethod
    def hydra_init(
        dists, dim, curvature, internals_exist=False, cv=0.01, cv_base="closest"
    ):
        """Initialise variational distributions using hydra+ and a coefficient of variation.
        Set the coefficient of variation base as either the 'closest' distance or 'norm'.
        """
        valid_cv_base = ("closest", "norm")
        if cv_base not in valid_cv_base:
            raise ValueError(f"Coefficient of variation must be in {valid_cv_base}")

        # embed tips with distances using HydraPlus
        hp_obj = hydraPlus.HydraPlus(dists, dim=dim, curvature=curvature)
        emm_tips = hp_obj.embed(equi_adj=0.0, alpha=1.1)
        print(
            "Embedding Stress (tips only) = {:.4}".format(emm_tips["stress_hydraPlus"])
        )
        leaf_loc_hyp = emm_tips["X"]

        if cv_base == "norm":
            # set variational parameters with small coefficient of variation
            # TODO: hyperboic norm
            leaf_sigma = np.abs(leaf_loc_hyp) * cv
            if internals_exist:
                int_loc_hyp = None
                int_sigma = None
        elif cv_base == "closest":
            # set leaf variational sigma using closest neighbour
            dists[dists == 0] = np.inf
            closest = dists.min(axis=0)
            closest = np.repeat([closest], dim, axis=0).transpose()
            leaf_sigma = np.abs(closest) * cv

        if internals_exist:
            param_init = {
                "leaf_mu": torch.from_numpy(leaf_loc_hyp).double(),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double(),
                "int_mu": torch.from_numpy(int_loc_hyp).double(),
                "int_sigma": torch.from_numpy(int_sigma).double(),
            }
        else:
            param_init = {
                "leaf_mu": torch.from_numpy(leaf_loc_hyp).double(),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double(),
            }
        return param_init

    @staticmethod
    def trace(epochs, path_write, elbo_hist):
        plt.figure()
        plt.plot(range(1, epochs), elbo_hist[1:], "r", label="elbo")
        plt.title("Elbo values")
        plt.xlabel("Epochs")
        plt.ylabel("elbo")
        plt.legend()
        plt.savefig(path_write + "/elbo_trace.png")

    def embed_tree_distribtution(self, dists=None):
        if dists is None:
            self.set_variationalParams_random()
        else:
            param_init = self.hydra_init(
                dists,
                self.D,
                self.curvature.numpy(),
                internals_exist=self.internals_exist,
            )
            self.set_variationalParams(param_init)

    def save(self, fn):
        with open(fn, "w", encoding="UTF-8") as f:
            f.write("Mix weights:\n")
            f.write("{self.VariationalParams['mix_weights'].detach().numpy()}\n\n")
            f.write("Leaf Locations (# taxa x  # dimensions):\n")
            for k in range(self.n_boosts):
                f.write(f"Mix {k}:\n")
                for i in range(self.S):
                    for d in range(self.D):
                        f.write("%f\t" % self.VariationalParams["leaf_mu"][k, i, d])
                    f.write("\n")
                f.write("\n")
            f.write("Leaf Sigmas (# taxa x  # dimensions):\n")
            for k in range(self.n_boosts):
                f.write(f"Mix {k}:\n")
                for i in range(self.S):
                    for d in range(self.D):
                        f.write("%f\t" % self.VariationalParams["leaf_sigma"][k, i, d])
                    f.write("\n")
                f.write("\n")

                if self.internals_exist:
                    f.write("Internal Locations (# taxa x  # dimensions):\n")
                    for i in range(self.S - 2):
                        for d in range(self.D):
                            f.write("%f\t" % self.VariationalParams["int_mu"][k, i, d])
                    f.write("Internal Sigmas (# taxa x  # dimensions):\n")
                    for i in range(self.S - 2):
                        for d in range(self.D):
                            f.write(
                                "%f\t" % self.VariationalParams["int_sigma"][k, i, d]
                            )
                        f.write("\n")
                    f.write("\n")

    def save_final_info(self, path_write, seconds):
        """Save time taken to info file."""
        file_name = path_write + "/" + "vi.log"
        with open(file_name, "a", encoding="UTF-8") as file:
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            file.write(f"Total time: {int(hrs)}:{int(mins)}:{int(secs)}\n")


def read(path_read, internals=True):
    with open(path_read, "r", encoding="UTF-8") as f:
        lines = [line.rstrip("\n") for line in f]
    dim = int(len([float(i) for i in lines[0].rstrip().split("\t")]) / 2)
    n_lines = len(lines) - 1
    if internals:
        n_taxa = int(n_lines / 2 + 1)
    else:
        n_taxa = n_lines

    VariationalParams = {
        "leaf_mu": np.empty((n_taxa, dim)),
        "leaf_sigma": np.empty((n_taxa, dim)),
    }

    for i in range(n_taxa):
        line_in = np.array([float(j) for j in lines[i].rstrip().split("\t")])
        VariationalParams["leaf_mu"][i, :] = line_in[:dim]
        VariationalParams["leaf_sigma"][i, :] = line_in[dim:]

    if internals:
        VariationalParams["int_mu"] = np.empty((n_taxa - 2, dim))
        VariationalParams["int_sigma"] = np.empty((n_taxa - 2, dim))
        for i in range(n_taxa - 2):
            line_in = np.array(
                [float(j) for j in lines[i + n_taxa].rstrip().split("\t")]
            )
        VariationalParams["int_mu"][i, :] = line_in[:dim]
        VariationalParams["int_sigma"][i, :] = line_in[dim:]
    return VariationalParams
