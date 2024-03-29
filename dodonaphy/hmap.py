"""Maximum A Posteriori Module on hyperboloid sheet."""
import os
import time

import numpy as np
import torch

from dodonaphy import Chyp_torch, peeler, tree, Cutils, Chyp_np
from dodonaphy.base_model import BaseModel


class HMAP(BaseModel):
    """Maximum A Posteriori class of embedding on hyperboloid sheet.

    Given a sequence alignment embed the tips onto the hyperboloid model of
    hyperbolic space, then optimise the tree likelihood of the decoded tree.

    """

    def __init__(
        self,
        partials,
        weights,
        dim,
        soft_temp,
        loss_fn,
        path_write,
        curvature=-1.0,
        prior="None",
        tip_labels=None,
        matsumoto=False,
        connector="nj",
        peel=None,
        normalise_leaves=False,
        model_name="JC69",
        freqs=None,
        embedder="wrap",
    ):
        self.path_write = path_write
        super().__init__(
            "hmap",
            partials,
            weights,
            dim=dim,
            soft_temp=soft_temp,
            connector=connector,
            curvature=torch.tensor(curvature),
            loss_fn=loss_fn,
            tip_labels=tip_labels,
            model_name=model_name,
            freqs=freqs,
            embedder=embedder,
        )
        self.normalise_leaves = normalise_leaves
        self.current_epoch = 0
        self.prior = prior
        self.peel = peel
        self.matsumoto = matsumoto
        self.params_optim = {}
        self.init_model_params()
        self.best_posterior = torch.tensor(-np.inf)
        self.best_freqs = self.phylomodel.freqs
        self.best_sub_rates = self.phylomodel.sub_rates

    def init_embedding_params(
        self, location_file=None, dists=None, hydra_max_iter=1000
    ):
        is_locations = location_file is not None
        is_dists = dists is not None
        if not is_locations and not is_dists:
            raise ValueError(
                "Must provide locations or distances to init HMAP, got neither."
            )
        if is_locations and is_dists:
            raise ValueError("Cannot provide both embedding locations and distances.")
        if is_locations:
            self.embed_locations(location_file)
        elif is_dists:
            self.embed_dists(dists, hydra_max_iter)

    def embed_locations(self, location_file):
        locs = self.read_embedding_base(location_file)
        if locs.shape[1] != self.D:
            raise ValueError(
                f"Embedding dimension {self.D} doesn't match data in file {location_file}"
            )
        if locs.shape[0] != self.S:
            raise ValueError(
                f"Number of taxa mishmatch: in alignment {self.S} and in is embedding {locs.shape[0]}"
            )
        self.params_optim["leaf_loc"] = torch.tensor(
            locs, requires_grad=True, dtype=torch.float64
        )
        self.log(f"Sucessfully read locations from file {location_file}\n")

    def embed_dists(self, dists, hydra_max_iter):
        # embed distances with hydra+
        emm_tips = self.hydra_init(dists, hydra_max_iter=hydra_max_iter)

        # set locations as parameters to optimise
        if self.normalise_leaves:
            if self.embedder == "wrap":
                raise NotImplementedError
            radius = np.mean(np.linalg.norm(emm_tips["X"], axis=1))
            directionals = Cutils.normalise_np(emm_tips["X"])
            self.params_optim["radius"] = torch.tensor(
                radius, requires_grad=True, dtype=torch.float64
            )
            self.params_optim["directionals"] = torch.tensor(
                directionals, requires_grad=True, dtype=torch.float64
            )
        else:
            if self.embedder == "wrap":
                # hydra+ uses vertical projection, need to adjust
                locs_hyp = Chyp_np.project_up_2d(emm_tips["X"])
                emm_tips["X"] = Chyp_np.unwrap_2d(locs_hyp)
            self.params_optim["leaf_loc"] = torch.tensor(
                emm_tips["X"], requires_grad=True, dtype=torch.float64
            )

    def learn(self, epochs, learn_rate, save_locations, start=""):
        """Optimise params["dists"].

        NB: start is just a string for printing: which tree was used to
        generate the original distance matrix.

        """
        start_time = time.time()
        self.loss = self.compute_loss()
        post_hist = [self.ln_p.item() + self.ln_prior.item()]
        self.record_if_best()

        def lr_lambda(epoch):
            return 1.0 / (epoch + 1.0) ** 0.25

        # Consider using LBFGS, but appears to not perform as well.
        optimizer = torch.optim.Adam(
            params=list(self.params_optim.values()), lr=learn_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #   optimizer, base_lr=learn_rate/100, max_lr=learn_rate, step_size_up=100)

        if self.path_write is not None:
            self.save(epochs, learn_rate, start)

        def closure():
            optimizer.zero_grad()
            self.loss = self.compute_loss()
            self.loss.backward()
            return self.loss

        print(f"Running for {epochs} iterations.")
        print("Iteration: log prior + log_likelihood = log posterior")
        for i in range(1, epochs + 1):
            self.current_epoch = i
            optimizer.step(closure)
            scheduler.step()
            self.record_if_best()
            self.print_epoch(i, post_hist)
            if self.path_write is not None:
                self.save_epoch(i, save_locations=save_locations)
            self.save_full_state("checkpoint.json")

        print(
            f"\nBest tree log posterior joint found: {self.best_posterior.item():.3f}"
        )
        self.save_duration(start_time)
        self.save_best_tree()
        self.save_embedding()
        if epochs > 0:
            self.log(f"Best log likelihood: {self.best_ln_p}\n")
            self.log(f"Best log prior: {self.best_ln_prior.item()}\n")

        if self.path_write is not None:
            self.trace(self.path_write, post_hist, label="likelihood")

    def save_embedding(self):
        if self.path_write is not None:
            filename = os.path.join(self.path_write, "embedding.txt")
            locs = self.best_locs
            self.save_embedding_base(filename, locs)
            self.log(f"Best embedding locations saved to {filename}\n")

    def compute_loss(self):
        if self.connector in ("nj") or self.loss_fn in ("pair_likelihood", "hypHC"):
            self.peel, self.blens, self.pdm = self.connect(get_pdm=True)
        else:
            self.peel, self.blens = self.connect(get_pdm=False)
        self.ln_prior = self.compute_ln_prior()
        self.ln_p = self.compute_ln_likelihood()
        return -self.ln_prior - self.ln_p

    def print_epoch(self, iteration, posterior_history):
        posterior_history.append(self.ln_p.item() + self.ln_prior.item())
        print(
            f"{iteration}: {self.ln_prior.item():.3f} + {self.ln_p.item():.3f} = {posterior_history[-1]:.3f}"
        )
        if (iteration) % 10 == 9:
            print()

    def save_best_tree(self):
        """Save the model and tree from the best iteration."""
        if self.path_write is not None:
            # save the best model
            self.phylomodel.freqs = self.best_freqs
            self.phylomodel.sub_rates = self.best_sub_rates
            file_model = os.path.join(
                self.path_write, f"{self.inference_name}_model.log"
            )
            self.phylomodel.save(file_model)
            # save the best tree
            tree.save_tree_head(
                self.path_write, "mape", self.tip_labels, translate=False
            )
            tree.save_tree(
                self.path_write,
                "mape",
                self.best_peel,
                self.best_blens,
                self.best_epoch,
                self.best_ln_p.item(),
                self.best_ln_prior.item(),
                self.name_id,
                last_tree=True,
            )
            self.log(f"Best curvature: {self.best_curvature.item()}\n")

    def save_duration(self, start_time):
        if self.path_write is not None:
            seconds = time.time() - start_time
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            self.log(f"Total time: {int(hrs)}:{int(mins)}:{int(secs)}\n")

    def record_if_best(self):
        if self.loss < -self.best_posterior or self.current_epoch == 0:
            self.best_posterior = -self.loss.detach().clone()
            self.best_ln_p = self.ln_p.detach().clone()
            self.best_ln_prior = self.ln_prior.detach().clone()
            self.best_peel = self.peel
            self.best_blens = self.blens.detach().clone()
            self.best_epoch = self.current_epoch
            self.best_freqs = self.phylomodel.freqs.detach().clone()
            self.best_sub_rates = self.phylomodel.sub_rates.detach().clone()
            self.best_curvature = self.curvature.detach().clone()
            self.best_locs = self.get_locs().detach().clone()

    def save(self, epochs, learn_rate, start):

        self.log("%-12s: %i\n" % ("# epochs", epochs))
        self.log("%-12s: %i\n" % ("Curvature", self.curvature))
        self.log("%-12s: %i\n" % ("Matsumoto", self.matsumoto))
        self.log("%s: %i\n" % ("Normalise Leaf", self.normalise_leaf))
        self.log("%-12s: %i\n" % ("Dimensions", self.D))
        self.log("%-12s: %i\n" % ("# Taxa", self.S))
        self.log("%-12s: %i\n" % ("# Patterns", self.L))
        self.log("%-12s: %f\n" % ("Learn Rate", learn_rate))
        self.log("%-12s: %f\n" % ("Soft temp", self.soft_temp))
        self.log("%-12s: %s\n" % ("Embed Mthd", self.embedder))
        self.log("%-12s: %s\n" % ("Connect Mthd", self.connector))
        self.log("%-12s: %s\n" % ("Loss function", self.loss_fn))
        self.log("%-12s: %s\n" % ("Prior", self.prior))
        self.log("%-12s: %s\n" % ("Start Tree", start))

    def save_epoch(self, i, save_locations=False):
        "Save posterior value and leaf locations to file."
        path_post = os.path.join(self.path_write, "posterior.txt")
        ln_p = self.ln_p.item()
        ln_prior = self.ln_prior.item()
        ln_post = ln_p + ln_prior
        if not os.path.isfile(path_post):
            with open(path_post, "w", encoding="UTF-8") as file:
                file.write("log prior, log likelihood, log posterior\n")
        with open(path_post, "a", encoding="UTF-8") as file:
            file.write(f"{ln_prior}, {ln_p}, {ln_post}\n")

        emm_path = os.path.join(self.path_write, "location")
        if save_locations:
            if not os.path.isdir(emm_path):
                os.mkdir(emm_path)
            emm_fn = os.path.join(emm_path, f"location_{i}.csv")
            print_header = "".join([f"dim{i}, " for i in range(self.D)])
            locs = self.get_locs().detach().numpy()
            np.savetxt(
                emm_fn,
                locs,
                delimiter=", ",
                header=print_header,
            )
        tree.save_tree(
            self.path_write,
            "samples",
            self.peel,
            self.blens,
            i,
            self.ln_prior,
            self.ln_p,
            self.name_id,
        )

    def get_locs(self):
        """Get current tip locations"""
        if self.normalise_leaves:
            locs = self.params_optim["radius"] * self.params_optim["directionals"]
        else:
            locs = self.params_optim["leaf_loc"]
        return locs

    def connect(self, get_pdm=False):
        """Connect tips into a tree"""
        locs = self.get_locs()

        if get_pdm or self.connector in ("nj"):
            pdm = Chyp_torch.get_pdm(
                locs, curvature=self.curvature, projection=self.embedder
            )

        if self.connector == "geodesics":
            peel, _, blens = peeler.make_soft_peel_tips(
                locs, connector="geodesics", curvature=self.curvature
            )
        elif self.connector == "nj":
            peel, blens = peeler.nj_torch(pdm, tau=self.soft_temp)
        elif self.connector == "fix":
            peel = self.peel
            blens = self.get_blens_fix_topo(locs)
        else:
            raise ValueError(
                f"Connection must be one of 'nj', 'geodesics', 'fix'. Got {self.connector}"
            )
        if get_pdm:
            return peel, blens, pdm
        return peel, blens

    def get_blens_fix_topo(self, locs):
        """Get the branch lengths from a fixed topology."""
        node_count = locs.shape[0]
        blens = torch.zeros(node_count, dtype=torch.double)
        k = 0
        for (l, r, p) in self.peel:
            if p == len(blens):
                pair_locs = torch.vstack((locs[l, :], locs[r, :]))
                blens[k] = Chyp_torch.get_pdm(
                    pair_locs, curvature=self.curvature, projection=self.embedder
                )[0, 1]
            else:
                pair_locs = torch.vstack((locs[l, :], locs[p, :]))
                blens[k] = Chyp_torch.get_pdm(
                    pair_locs, curvature=self.curvature, projection=self.embedder
                )[0, 1]
                k += 1
                pair_locs = torch.vstack((locs[r, :], locs[p, :]))
                blens[k] = Chyp_torch.get_pdm(
                    pair_locs, curvature=self.curvature, projection=self.embedder
                )[0, 1]
                k += 1
        return blens

    def compute_ln_likelihood(self):
        """Compute likelihood of current tree, reducing soft_temp as required."""
        self.ln_p = self.compute_LL(self.peel, self.blens)
        if self.loss_fn == "likelihood":
            loss = self.ln_p
        elif self.loss_fn == "pair_likelihood":
            loss = self.compute_log_a_like(self.pdm)
        elif self.loss_fn == "hypHC":
            locs = self.get_locs()
            loss = self.compute_likelihood_hypHC(
                self.pdm, locs, temperature=0.05, n_triplets=100
            )
        return loss

    def compute_ln_prior(self):
        if self.prior == "None":
            return torch.zeros(1)
        prior_sub_rates = self.phylomodel.compute_ln_prior_sub_rates()
        prior_freqs = self.phylomodel.compute_ln_prior_freqs()
        prior_tree = self.compute_ln_tree_prior()
        return prior_sub_rates + prior_freqs + prior_tree

    def compute_ln_tree_prior(self):
        """Compute prior of current tree."""
        if self.prior == "None":
            return torch.zeros(1)
        elif self.prior == "normal":
            locs = self.get_locs()
            return self.compute_prior_normal(locs)
        elif self.prior == "uniform":
            locs = self.get_locs()
            return self.compute_prior_unif(locs, scale=10.0)
        elif self.prior == "gammadir":
            return self.compute_prior_gamma_dir(self.blens)
        elif self.prior == "birthdeath":
            raise ValueError(
                "Birth death model implementation isn't differentiable. It cannot be used for gradient based map estimation."
            )
