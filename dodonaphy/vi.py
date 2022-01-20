import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import hydraPlus, tree, utils
from .base_model import BaseModel
from .phylo import JC69_p_t, calculate_treelikelihood


class DodonaphyVI(BaseModel):
    def __init__(
        self,
        partials,
        weights,
        dim,
        embedder="wrap",
        connector="mst",
        curvature=-1.0,
        soft_temp=None,
        noise=None,
        truncate=None,
    ):
        super().__init__(
            partials,
            weights,
            dim,
            soft_temp=soft_temp,
            embedder=embedder,
            connector=connector,
            curvature=curvature,
        )
        print("Initialising variational model.\n")

        # Store mu on poincare ball in R^dim.
        # Distributions stored in tangent space T_0 H^D, then transformed to poincare ball.
        # The distribution for each point has a single sigma (i.e. mean field in x, y).
        eps = np.finfo(np.double).eps
        leaf_sigma = np.log(np.abs(np.random.randn(self.S, self.D)) + eps)
        int_sigma = np.log(np.abs(np.random.randn(self.S - 2, self.D)) + eps)
        self.noise = noise
        self.truncate = truncate
        self.ln_p = self.compute_LL(self.peel, self.blens)

        if self.connector in ("geodesics", "nj"):
            self.VariationalParams = {
                "leaf_mu": torch.randn(
                    (self.S, self.D), requires_grad=True, dtype=torch.float64
                ),
                "leaf_sigma": torch.tensor(
                    leaf_sigma, requires_grad=True, dtype=torch.float64
                ),
            }
        elif self.connector == "mst":
            self.VariationalParams = {
                "leaf_mu": torch.randn(
                    (self.S, self.D), requires_grad=True, dtype=torch.float64
                ),
                "leaf_sigma": torch.tensor(
                    leaf_sigma, requires_grad=True, dtype=torch.float64
                ),
                "int_mu": torch.randn(
                    (self.S - 2, self.D), requires_grad=True, dtype=torch.float64
                ),
                "int_sigma": torch.tensor(
                    int_sigma, requires_grad=True, dtype=torch.float64
                ),
            }

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
                leaf_loc = self.VariationalParams["leaf_mu"].reshape(n_tip_params)
                leaf_cov = torch.eye(
                    n_tip_params, dtype=torch.double
                ) * self.VariationalParams["leaf_sigma"].exp().reshape(n_tip_params)
                if self.connector == "mst":
                    n_int_params = torch.numel(self.VariationalParams["int_mu"])
                    int_loc = self.VariationalParams["int_mu"].reshape(n_int_params)
                    int_cov = torch.eye(
                        n_int_params, dtype=torch.double
                    ) * self.VariationalParams["int_sigma"].exp().reshape(n_int_params)
                    sample = self.sample(leaf_loc, leaf_cov, int_loc, int_cov)
                else:
                    sample = self.sample(leaf_loc, leaf_cov)

                peel.append(sample["peel"])
                blens.append(sample["blens"])
                if self.connector == "mst":
                    location.append(
                        utils.dir_to_cart_tree(
                            sample["leaf_r"],
                            sample["int_r"],
                            sample["leaf_dir"],
                            sample["int_dir"],
                            self.D,
                        )
                    )
                else:
                    location.append(
                        utils.dir_to_cart(sample["leaf_r"], sample["leaf_dir"])
                    )
                if kwargs.get("lp"):
                    LL = calculate_treelikelihood(
                        self.partials,
                        self.weights,
                        sample["peel"],
                        JC69_p_t(sample["blens"]),
                        torch.full([4], 0.25, dtype=torch.float64),
                    )
                    lp.append(LL)

        if kwargs.get("lp"):
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
        leaf_cov = torch.eye(n_tip_params, dtype=torch.double) * self.VariationalParams[
            "leaf_sigma"
        ].exp().reshape(n_tip_params)
        if self.connector == "mst":
            n_int_params = torch.numel(self.VariationalParams["int_mu"])
            int_loc = self.VariationalParams["int_mu"].reshape(n_int_params)
            int_cov = torch.eye(
                n_int_params, dtype=torch.double
            ) * self.VariationalParams["int_sigma"].exp().reshape(n_int_params)
            sample = self.sample(leaf_loc, leaf_cov, int_loc, int_cov)
        else:
            sample = self.sample(leaf_loc, leaf_cov)

        # if self.connector == "mst":
        #     pdm = Chyperboloid.get_pdm_torch(
        #         sample["leaf_r"].repeat(self.S),
        #         sample["leaf_dir"],
        #         sample["int_r"],
        #         sample["int_dir"],
        #         curvature=self.curvature,
        #     )
        # else:
        #     pdm = Chyperboloid.get_pdm_torch(
        #         sample["leaf_r"].repeat(self.S),
        #         sample["leaf_dir"],
        #         curvature=self.curvature,
        #     )
        # logPrior = self.compute_prior_gamma_dir(sample['blens'])
        # logP = self.compute_LL(sample['peel'], sample['blens'])
        # logPrior = self.compute_prior_gamma_dir(pdm[:])
        # anneal_epoch = torch.tensor(100)
        # min_temp = torch.tensor(.1)
        # temp = torch.maximum(torch.exp(- self.epoch / anneal_epoch), min_temp)
        # logP = self.compute_log_a_like(pdm, temp=temp)

        return sample["ln_p"] + sample["ln_prior"] - sample["logQ"] + sample["jacobian"]

    def learn(
        self, param_init=None, epochs=1000, k_samples=3, path_write="./out", lr=1e-3
    ):
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
            if self.connector == "mst":
                self.VariationalParams["int_mu"] = param_init["int_mu"]
                self.VariationalParams["int_sigma"] = param_init["int_sigma"]

        if path_write is not None:
            fn = path_write + "/" + "vi.info"
            with open(fn, "w", encoding="UTF-8") as file:
                file.write("%-12s: %i\n" % ("# epochs", epochs))
                file.write("%-12s: %i\n" % ("k_samples", k_samples))
                file.write("%-12s: %i\n" % ("Dimensions", self.D))
                file.write("%-12s: %i\n" % ("# Taxa", self.S))
                file.write("%-12s: %i\n" % ("Patterns", self.L))
                file.write("%-12s: %f\n" % ("Learn Rate", lr))
                file.write("%-12s: %s\n" % ("Embed Mthd", self.embedder))
                file.write("%-12s: %s\n" % ("Connect Mthd", self.connector))

            vi_path = os.path.join(path_write, "vi_params")
            os.mkdir(vi_path)
            fn = os.path.join(vi_path, f"vi_{0}.csv")
            self.save(fn)

            elbo_fn = os.path.join(path_write, "elbo.txt")

        def lr_lambda(epoch):
            return 1.0 / np.sqrt(epoch + 1)

        optimizer = torch.optim.Adam(list(self.VariationalParams.values()), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        hist_dat: List[Any] = []

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = -self.elbo_normal(k_samples)
            if loss.requires_grad:
                loss.backward()
            elbo_hist.append(-loss.item())
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
            scheduler.step()
            print("epoch %-12i ELBO: %10.3f" % (epoch + 1, elbo_hist[-1]))
            hist_dat.append(elbo_hist[-1])

            if path_write is not None:
                with open(elbo_fn, "a", encoding="UTF-8") as f:
                    f.write("%f\n" % elbo_hist[-1])
                fn = os.path.join(path_write, "vi_params", f"vi_{epoch+1}.csv")
                self.save(fn)

        if epochs > 0 and path_write is not None:
            self.trace(epochs, path_write, hist_dat, elbo_hist)

        if path_write is not None:
            final_elbo = self.elbo_normal(100).item()
            print("Final ELBO: {}".format(final_elbo))
            fn = os.path.join(path_write, "vi.info")
            with open(fn, "a", encoding="UTF-8") as file:
                file.write("%-12s: %i\n" % ("Final ELBO (100 samples)", final_elbo))

    @staticmethod
    def trace(epochs, path_write, hist_dat, elbo_hist):
        plt.figure()
        plt.plot(range(1, epochs), elbo_hist[1:], "r", label="elbo")
        plt.title("Elbo values")
        plt.xlabel("Epochs")
        plt.ylabel("elbo")
        plt.legend()
        plt.savefig(path_write + "/elbo_trace.png")

        plt.clf()
        plt.hist(hist_dat)
        plt.savefig(path_write + "/elbo_hist.png")

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
    def run(
        dim,
        S,
        partials,
        weights,
        dists_data,
        path_write,
        epochs=1000,
        k_samples=3,
        n_draws=100,
        n_grids=10,
        n_trials=100,
        max_scale=1,
        embedder="wrap",
        lr=1e-3,
        curvature=-1.0,
        connector="nj",
        soft_temp=None,
    ):
        """Initialise and run Dodonaphy's variational inference

        Initialise the emebedding with tips distances given to hydra+.
        Internal nodes are in distributions at origin.

        """
        print("\nRunning Dodonaphy Variational Inference.")
        print("Using %s embedding with %s connections" % (embedder, connector))

        # embed tips with distances using HydraPlus
        hp_obj = hydraPlus.HydraPlus(dists_data, dim=dim, curvature=curvature)
        emm_tips = hp_obj.embed(equi_adj=0.0, stress=True)
        print("Embedding Stress (tips only) = {:.4}".format(emm_tips["stress"].item()))

        # Initialise model
        mymod = DodonaphyVI(
            partials,
            weights,
            dim,
            embedder=embedder,
            connector=connector,
            soft_temp=soft_temp,
            curvature=curvature,
        )

        # Choose internal node locations from best random initialisation
        if connector == "mst":
            int_r, int_dir = mymod.initialise_ints(
                emm_tips, n_grids=n_grids, n_trials=n_trials, max_scale=max_scale
            )

        # convert to cartesian coords
        leaf_loc_poin = utils.dir_to_cart(
            torch.from_numpy(emm_tips["r"]), torch.from_numpy(emm_tips["directional"])
        )
        if connector == "mst":
            int_loc_poin = torch.from_numpy(utils.dir_to_cart(int_r, int_dir))

        # set variational parameters with small coefficient of variation
        cv = 1.0 / 100
        eps = np.finfo(np.double).eps
        # leaf_sigma = np.log(np.abs(np.array(leaf_loc_poin)) * cv + eps)
        if connector == "mst":
            int_sigma = np.log(np.abs(np.array(int_loc_poin)) * cv + eps)

        # set leaf variational sigma using closest neighbour
        dists_data[dists_data == 0] = np.inf
        closest = dists_data.min(axis=0)
        closest = np.repeat([closest], dim, axis=0).transpose()
        leaf_sigma = np.log(np.abs(closest) * cv + eps)

        if connector == "mst":
            param_init = {
                "leaf_mu": leaf_loc_poin.clone().double().requires_grad_(True),
                "leaf_sigma": torch.from_numpy(leaf_sigma)
                .double()
                .requires_grad_(True),
                "int_mu": int_loc_poin.clone().double().requires_grad_(True),
                "int_sigma": torch.from_numpy(int_sigma).double().requires_grad_(True),
            }
        elif connector in ("geodesics", "nj"):
            param_init = {
                "leaf_mu": leaf_loc_poin.clone().double().requires_grad_(True),
                "leaf_sigma": torch.from_numpy(leaf_sigma)
                .double()
                .requires_grad_(True),
            }

        # learn ML
        # mymod.learn_ML_brute(param_init=param_init)

        # learn
        mymod.learn(
            param_init=param_init,
            epochs=epochs,
            k_samples=k_samples,
            path_write=path_write,
            lr=lr,
        )

        # draw samples (one-by-one for reduced memory requirements)
        # and save them
        if path_write is not None:
            tree.save_tree_head(path_write, "vi", S)
            for i in range(n_draws):
                peels, blens, _, lp = mymod.draw_sample(1, lp=True)
                ln_prior = mymod.compute_prior_gamma_dir(blens[0])
                tree.save_tree(
                    path_write,
                    "vi",
                    peels[0],
                    blens[0],
                    i,
                    lp[0].item(),
                    ln_prior.item(),
                )

    def save(self, fn):
        with open(fn, "w", encoding="UTF-8") as f:
            for i in range(self.S):
                for d in range(self.D):
                    f.write("%f\t" % self.VariationalParams["leaf_mu"][i, d])
                for d in range(self.D):
                    f.write("%f\t" % self.VariationalParams["leaf_sigma"][i, d])
                f.write("\n")
            f.write("\n")

            if self.connector == "mst":
                for i in range(self.S - 2):
                    for d in range(self.D):
                        f.write("%f\t" % self.VariationalParams["int_mu"][i, d])
                    for d in range(self.D):
                        f.write("%f\t" % self.VariationalParams["int_sigma"][i, d])
                    f.write("\n")
                f.write("\n")


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
