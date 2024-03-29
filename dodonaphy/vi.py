import os
import time
import warnings
from typing import Any, List

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.transforms import StickBreakingTransform

from dodonaphy import Chyp_torch, peeler, tree, utils, Chyp_np
from dodonaphy.base_model import BaseModel


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
        model_name="JC69",
        freqs=None,
        path_write=".",
        prior_fn="gammadir",
    ):
        super().__init__(
            "vi",
            partials,
            weights,
            dim,
            soft_temp=soft_temp,
            embedder=embedder,
            connector=connector,
            curvature=torch.tensor(curvature),
            tip_labels=tip_labels,
            model_name=model_name,
            freqs=freqs,
        )
        print("Initialising variational model.\n")

        # Store distribution centres mu in hyperboloid projected onto R^dim.
        # The last coordinate in R^(dim+1) is determined.
        self.n_boosts = n_boosts
        self.path_write = path_write
        self.noise = noise
        self.truncate = truncate
        # Variational parameters must be set using set_params_optim() or set_params_optim_random()
        self.params_optim = dict()
        # set evolutionary model parameters to optimise
        self.init_model_params()
        self.simplex_transform = StickBreakingTransform()
        
        assert prior_fn in ("exponential", "gammadir"), "Invalid prior specified."
        self.prior_fn = prior_fn

        if self.connector == "fix":
            raise NotImplementedError(
                "Tree connection cannot yet be fixed. An easy TODO."
            )

    def set_params_optim_random(self):
        exp_distr = torch.distributions.Exponential(1.0)
        leaf_sigma = exp_distr.sample((self.n_boosts, self.S, self.D))
        int_sigma = exp_distr.sample((self.n_boosts, self.S - 2, self.D))

        self.params_optim = {
            "leaf_mu": torch.randn(
                (self.n_boosts, self.S, self.D),
                requires_grad=True,
            ),
            "leaf_sigma": leaf_sigma.requires_grad_(True)
        }
        if self.n_boosts > 1:
            self.mix_weights = torch.full((self.n_boosts,), 1 / self.n_boosts)
            self.params_optim["mix_weights"] = self._mix_weights
        else:
            self.mix_weights = torch.ones(1)
        
        if self.internals_exist:
            self.params_optim["int_mu"] = torch.randn(
                (self.n_boosts, self.S - 2, self.D),
                requires_grad=True
            )
            self.params_optim["int_sigma"] = int_sigma.requires_grad_(True)

    def set_params_optim(self, param_init):
        """Set variational parameters to optimise

        Args:
            param_init (Dict): A dictionary containing:
            leaf_mu     - node locations
            leaf_sigma  - standard deviation of locations
            int_mu      - internal node locations
            int_signma  - internal node sd of locations
            mix_weights - mixture weights
        """
        # set dimensions of input
        if param_init["leaf_mu"].ndim == 2:
            param_init["leaf_mu"].unsqueeze(0)
        if param_init["leaf_sigma"].ndim == 2:
            param_init["leaf_sigma"].unsqueeze(0)
        # set leaf mean locations
        self.params_optim["leaf_mu"] = (
            param_init["leaf_mu"].repeat((self.n_boosts, 1, 1)).requires_grad_()
        )
        # set leaf scale (sigma in normal distribution)
        self.params_optim["leaf_sigma"] = (
            param_init["leaf_sigma"].repeat((self.n_boosts, 1, 1)).requires_grad_()
        )
        if self.internals_exist:
            if param_init["int_mu"].ndim == 2:
                param_init["int_mu"].unsqueeze(0)
            if param_init["int_sigma"].ndim == 2:
                param_init["int_sigma"].unsqueeze(0)
            self.params_optim["int_mu"] = (
                param_init["int_mu"].repeat((self.n_boosts, 1, 1)).requires_grad_()
            )
            self.params_optim["int_sigma"] = (
                param_init["int_sigma"].repeat((self.n_boosts, 1, 1)).requires_grad_()
            )

        if "mix_weights" in param_init.keys():
            # set weights specifically
            self.mix_weights = param_init["mix_weights"]
        elif self.n_boosts > 1:
            # otherwise set equal weights
            self.mix_weights = torch.full(
                (self.n_boosts,),
                1.0 / self.n_boosts,
                dtype=torch.float64,
            )
        else:
            # default to 1 mixture
            self.mix_weights = torch.ones(1)
        # optimise the mixture weights
        if self.n_boosts > 1:
            self.params_optim["mix_weights"] = self._mix_weights
        # optimise the curvature
        # self.curvature = self.curvature.detach().clone()
        self.params_optim["curvature"] = self._curvature

    @property
    def mix_weights(self):
        return self.simplex_transform(self._mix_weights)

    @mix_weights.setter
    def mix_weights(self, mix_weights):        
        if not hasattr(self, "_mix_weights"):
            self._mix_weights = self.simplex_transform.inv(mix_weights).clone().detach().requires_grad_(True)
        else:
            self._mix_weights = self.simplex_transform.inv(mix_weights)

    def calculate_elbo(self, mix_idx, path_write, file_name, iteration):
        """Calculate the elbo of a sample from the variational distributions q_k

        Args:
            mix_idx (int): index of mixture

        Returns:
            float: The evidence lower bound of a sample from q
        """
        n_tip_params = torch.numel(self.params_optim["leaf_mu"][mix_idx])
        leaf_locs = self.params_optim["leaf_mu"][mix_idx].reshape(n_tip_params)
        leaf_cov = torch.eye(n_tip_params, dtype=torch.double) * self.params_optim[
            "leaf_sigma"
        ][mix_idx].exp().reshape(n_tip_params)
        if self.internals_exist:
            n_int_params = torch.numel(self.params_optim["int_mu"][mix_idx])
            int_locs = self.params_optim["int_mu"][mix_idx].reshape(n_int_params)
            int_cov = torch.eye(n_int_params, dtype=torch.double) * self.params_optim[
                "int_sigma"
            ][mix_idx].exp().reshape(n_int_params)
            sample = self.rsample_tree(
                leaf_locs,
                leaf_cov,
                int_locs,
                int_cov,
                path_write=path_write,
                file_name=file_name,
                iteration=iteration,
            )
        else:
            sample = self.rsample_tree(
                leaf_locs,
                leaf_cov,
                path_write=path_write,
                file_name=file_name,
                iteration=iteration,
            )

        if sample["jacobian"] == -torch.inf:
            warnings.warn("Log Jacobian determinant set to zero.")
            sample["jacobian"] = 0.0
        return sample["ln_p"] + sample["ln_prior"] - sample["logQ"] + sample["jacobian"]

    def learn(
        self,
        epochs=1000,
        importance_samples=1,
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

        if self.path_write is not None:
            self.log_run_start(self.path_write, epochs, importance_samples, lr)
            self.elbo_fn = os.path.join(self.path_write, "elbo.txt")

        def lr_lambda(epoch):
            return 1.0 / np.sqrt(epoch + 1)

        # Consider using LBFGS, but appears to not perform as well.
        optimizer = torch.optim.Adam(list(self.params_optim.values()), lr=lr)
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

        self.log("\nLearning...\n")
        for epoch in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            print("epoch %-12i ELBO: %10.3f" % (epoch + 1, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

            if self.path_write is not None:
                self.log_elbo(elbo_hist[-1])
                fn = os.path.join(self.path_write, "vi_params", "latest.csv")
                self.save(fn)
                fn = os.path.join(self.path_write, "vi_params", f"iteration.txt")
                with open(fn, "w", encoding="UTF-8") as file:
                    file.write(f"Epoch: {epoch + 1} / {epochs}")
            
                self.save_full_state(os.path.join(self.path_write, "checkpoint.json"))

        if self.path_write is not None:
            self.trace(self.path_write, elbo_hist, label="elbo")

        if self.path_write is not None:
            self.compute_final_elbo(self.path_write, n_draws)
            file_model = os.path.join(
                self.path_write, f"{self.inference_name}_model.log"
            )
            self.phylomodel.save(file_model)
            self.save_final_info(self.path_write, time.time() - start_time)
            self.log(f"Final curvature: {self.curvature.item()}\n")

    def compute_final_elbo(self, path_write, n_draws):
        # draw samples from the final distribution and save them
        file_name = "samples"
        tree.save_tree_head(path_write, file_name, self.tip_labels, translate=False)
        with torch.no_grad():
            final_elbo = -self.elbo_siwae(
                importance=n_draws, path_write=path_write, file_name=file_name
            ).item()
        tree.end_tree_file(path_write)
        self.log("%-12s: %f\n" % (f"Final ELBO ({n_draws} samples)", final_elbo))
        print(f"Final ELBO: {final_elbo:.3f}")

    def log_run_start(self, path_write, epochs, importance_samples, lr):
        self.log("\n Variational inference parameters:\n")
        self.log("%-12s: %i\n" % ("# epochs", epochs))
        self.log("%-12s: %s\n" % ("prior", "Gamma Dirichlet"))
        self.log("%-12s: %i\n" % ("Importance", importance_samples))
        self.log("%-12s: %i\n" % ("# mixtures", self.n_boosts))
        self.log("%-12s: %f\n" % ("Curvature", self.curvature))
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

        vi_path = os.path.join(path_write, "vi_params")
        os.makedirs(vi_path, exist_ok=True)
        fn = os.path.join(vi_path, "start.csv")
        self.save(fn)

    def log_elbo(self, elbo_value):
        with open(self.elbo_fn, "a", encoding="UTF-8") as f:
            f.write("%f\n" % elbo_value)

    def elbo_siwae(self, importance=1, path_write=None, file_name=None):
        """Compute the ELBO.

        Args:
            importance (int, optional): Number of importance samples of elbo (IWAE).
            Defaults to 1.
            ln_elbos (tensor, optional): Provide the precomputed log elbo values. Defaults to None

        Returns:
            [torch.Tensor]: ELBO value
        """
        ln_elbos = torch.zeros((importance, self.n_boosts))
        sample_number = 0
        for k in range(self.n_boosts):
            for t in range(importance):
                ln_elbos[t, k] = self.calculate_elbo(
                    k, path_write, file_name, sample_number
                )
                sample_number += 1
        mixture_logit = torch.log_softmax(self.mix_weights, dim=0)
        loss_n = torch.logsumexp(
            -torch.log(torch.tensor(importance))
            + mixture_logit
            + ln_elbos,
            dim=1,
        )
        return loss_n.mean(dim=0)

    def rsample_tree(
        self,
        leaf_locs,
        leaf_cov,
        normalise_leaf=False,
        path_write=None,
        file_name=None,
        iteration=None,
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
        sample_leaf_locs, log_Q = self.rsample_Euclid(
            leaf_locs, leaf_cov, is_internal=False, normalise_loc=normalise_leaf
        )
        # transform into tree
        peel, blens, pdm = self.connect(sample_leaf_locs)

        # get jacobian
        def get_blens(locs_t0):
            locs_t0_2d = locs_t0.reshape((self.S, self.D))
            _, blens, _ = self.connect(locs_t0_2d)
            return blens

        # TODO: use analytical form
        jacobian = torch.autograd.functional.jacobian(
            get_blens, sample_leaf_locs.flatten()
        )
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

        # save sample
        if (path_write is not None) != (file_name is not None):
            raise ValueError(
                f"path {path_write} and file name {file_name} should either both be None or a string"
            )
        can_save = path_write is not None and file_name is not None
        can_save *= isinstance(path_write, str)
        can_save *= isinstance(file_name, str)
        if can_save:
            tree.save_tree(
                path_write,
                file_name,
                peel,
                blens,
                iteration,
                ln_p,
                ln_prior,
                self.name_id,
                lnQ = log_Q,
                lnJac = log_abs_det_jacobian,
            )

        return sample

    def get_loss(self, peel, blens, pdm, leaf_locs):
        if self.loss_fn == "likelihood":
            ln_p = self.compute_LL(peel, blens)
        elif self.loss_fn == "pair_likelihood":
            ln_p = self.compute_log_a_like(pdm)
        elif self.loss_fn == "hypHC":
            ln_p = self.compute_hypHC(pdm, leaf_locs)

        if self.prior_fn == "exponential":
            ln_prior = self.compute_prior_exponential(blens, rate=10.0)
        elif self.prior_fn == "gammadir":
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
        pdm = Chyp_torch.get_pdm(
            locs_t0, curvature=self.curvature, projection=self.embedder
        )
        if self.connector == "geodesics":
            peel, int_locs, blens = peeler.make_soft_peel_tips(
                locs_t0, connector="geodesics", curvature=self.curvature
            )
        elif self.connector == "nj":
            peel, blens = peeler.nj_torch(pdm, tau=self.soft_temp)
        elif self.connector == "nj-r":
            raise ValueError("No gradient available from rapid NJ.")
        elif self.connector == "fix":
            raise NotImplementedError(
                "Variational inference on fixed topology. Need internal nodes."
            )
        return peel, blens, pdm

    def embed_dists(self, dists, hydra_max_iter=0):
        """Initialise variational distributions using hydra+ and a coefficient of variation.
        Set the coefficient of variation base as either the 'closest' distance or 'norm'.
        """
        # embed tips with distances using HydraPlus
        emm_tips = self.hydra_init(dists, hydra_max_iter=hydra_max_iter)
        leaf_loc_hyp = emm_tips["X"]
        if self.embedder == "wrap":
            locs_hyp = Chyp_np.project_up_2d(emm_tips["X"])
            emm_tips["X"] = Chyp_np.unwrap_2d(locs_hyp)

        leaf_sigma = self.get_sigma(leaf_loc_hyp, dists=dists)
        param_init = self.set_init_q(leaf_loc_hyp, leaf_sigma)
        return param_init

    def get_sigma(self, leaf_loc_hyp, dists=None, cv=0.01, cv_base="closest"):
        valid_cv_base = ("closest", "norm")
        if cv_base not in valid_cv_base:
            raise ValueError(f"Coefficient of variation must be in {valid_cv_base}")
        if cv_base == "norm":
            # set variational parameters with small coefficient of variation
            leaf_sigma = np.abs(leaf_loc_hyp) * cv
        elif cv_base == "closest":
            # set leaf variational sigma using closest neighbour
            dists[dists == 0.0] = np.inf
            closest = dists.min(axis=0)
            closest = np.repeat([closest], self.D, axis=0).transpose()
            leaf_sigma = np.abs(closest) * cv
            dists[dists == np.inf] = 0.0
        return leaf_sigma

    def set_init_q(self, leaf_loc, leaf_sigma, int_loc=None, int_sigma=None):
        if self.internals_exist:
            param_init = {
                "leaf_mu": torch.from_numpy(leaf_loc).double(),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double(),
                "int_mu": torch.from_numpy(int_loc).double(),
                "int_sigma": torch.from_numpy(int_sigma).double(),
            }
        else:
            param_init = {
                "leaf_mu": torch.from_numpy(leaf_loc).double(),
                "leaf_sigma": torch.from_numpy(leaf_sigma).double(),
            }
        return param_init


    def embed_tree_distribtution(
        self, dists=None, location_file=None, hydra_max_iter=0
    ):
        use_locations = location_file is not None
        use_dists = dists is not None
        if use_locations and use_dists:
            raise ValueError("Only provide distances OR a location file to embed.")
        if use_locations:
            self.log(f"Starting tree disregarded.\n")
            self.log(f"Initialising embedding from location file.\n")
            param_init = self.set_params_file(location_file)
            self.set_params_optim(param_init)
        elif use_dists:
            self.log(f"Initialising embedding from tree distances.\n")
            param_init = self.embed_dists(dists, hydra_max_iter=hydra_max_iter)
            self.set_params_optim(param_init)
        else:
            self.set_params_optim_random()

    def set_params_file(self, location_file):
        locations = self.read_embedding_base(location_file)
        leaf_sigma = self.get_sigma(locations, cv_base="norm")
        param_init = self.set_init_q(locations, leaf_sigma)
        return param_init

    def save(self, fn):
        with open(fn, "w", encoding="UTF-8") as f:
            f.write("Mix weights:\n")
            f.write(f"{self.mix_weights.detach().numpy()}\n\n")
            f.write("Leaf Locations (# taxa x  # dimensions):\n")
            for k in range(self.n_boosts):
                f.write(f"Mix {k}:\n")
                for i in range(self.S):
                    for d in range(self.D):
                        f.write("%f\t" % self.params_optim["leaf_mu"][k, i, d])
                    f.write("\n")
                f.write("\n")
            f.write("Leaf Sigmas (# taxa x  # dimensions):\n")
            for k in range(self.n_boosts):
                f.write(f"Mix {k}:\n")
                for i in range(self.S):
                    for d in range(self.D):
                        f.write("%f\t" % self.params_optim["leaf_sigma"][k, i, d])
                    f.write("\n")
                f.write("\n")

                if self.internals_exist:
                    f.write("Internal Locations (# taxa x  # dimensions):\n")
                    for i in range(self.S - 2):
                        for d in range(self.D):
                            f.write("%f\t" % self.params_optim["int_mu"][k, i, d])
                    f.write("Internal Sigmas (# taxa x  # dimensions):\n")
                    for i in range(self.S - 2):
                        for d in range(self.D):
                            f.write("%f\t" % self.params_optim["int_sigma"][k, i, d])
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

    params_optim = {
        "leaf_mu": np.empty((n_taxa, dim)),
        "leaf_sigma": np.empty((n_taxa, dim)),
    }

    for i in range(n_taxa):
        line_in = np.array([float(j) for j in lines[i].rstrip().split("\t")])
        params_optim["leaf_mu"][i, :] = line_in[:dim]
        params_optim["leaf_sigma"][i, :] = line_in[dim:]

    if internals:
        params_optim["int_mu"] = np.empty((n_taxa - 2, dim))
        params_optim["int_sigma"] = np.empty((n_taxa - 2, dim))
        for i in range(n_taxa - 2):
            line_in = np.array(
                [float(j) for j in lines[i + n_taxa].rstrip().split("\t")]
            )
        params_optim["int_mu"][i, :] = line_in[:dim]
        params_optim["int_sigma"][i, :] = line_in[dim:]
    return params_optim
