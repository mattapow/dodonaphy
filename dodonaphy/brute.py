import matplotlib.pyplot as plt
import numpy as np
import torch

from dodonaphy import Chyp_torch, peeler, tree, utils, Cpeeler, Chyp_np
from dodonaphy.vi import DodonaphyVI


class Brute(DodonaphyVI):
    """Brute force learn an Embedding"""

    @staticmethod
    def run(
        dim,
        partials,
        weights,
        dists,
        path_write,
        epochs=1000,
        n_boosts=1,
        importance_samples=1,
        n_draws=100,
        embedder="wrap",
        lr=1e-3,
        curvature=-1.0,
        connector="nj",
        soft_temp=None,
        tip_labels=None,
    ):
        """Brute force algorithm to do grid search."""
        print("\nRunning Dodonaphy Brute Force Search.")
        print("Using %s embedding with %s connections" % (embedder, connector))

        # embed tips with distances using HydraPlus
        hp_obj = hydraPlus.HydraPlus(dists, dim=dim, curvature=curvature)
        emm_tips = hp_obj.embed(equi_adj=0.0, stress=True)
        print(
            "Embedding Stress (tips only) = {:.4}".format(emm_tips["stress_hydraPlus"])
        )

        # Initialise model
        mymod = DodonaphyVI(
            partials,
            weights,
            dim,
            embedder=embedder,
            connector=connector,
            soft_temp=soft_temp,
            curvature=curvature,
            tip_labels=tip_labels,
            n_boosts=n_boosts,
        )

        leaf_loc_hyp = emm_tips["X"]
        if mymod.internals_exist:
            int_loc_hyp = None
            param_init = {
                "leaf_loc": torch.from_numpy(leaf_loc_hyp).double(),
                "int_loc": torch.from_numpy(int_loc_hyp).double(),
            }
        else:
            param_init = {
                "leaf_loc": torch.from_numpy(leaf_loc_hyp).double(),
            }

        # mymod.learn_ML_brute(param_init=param_init)
        idx = 0
        mymod.grid_search_LL(idx, isLeaf=True, doPlot=True)

    def embedding_LL(self, int_loc_optim):
        leaf_loc = self.VariationalParams["leaf_mu"]
        int_loc = self.VariationalParams["int_mu"]
        with torch.no_grad():
            int_loc[0, :] = int_loc_optim
        assert self.connector == "simple"
        leaf_poin = utils.real2ball(leaf_loc)
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)

        int_poin = utils.real2ball(int_loc)
        int_r, int_dir = utils.cart_to_dir(int_poin)

        # NB: peel_np has no gradient. Gradients through branch lengths only
        peel_np = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
        blen = self.compute_branch_lengths(
            self.S, peel_np, leaf_r, leaf_dir, int_r, int_dir, useNP=False
        )
        return self.compute_LL(peel_np, blen)

    # def learn_ML(self, iter_per_param=500, rounds=2, path_write='./out', lr=.1):
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    #     leaf_loc = leaf_loc.requires_grad_(True)
    #     int_loc = int_loc.requires_grad_(True)
    #     from torchviz import make_dot
    #     for iter in range(iter_per_param):
    #     optimizer.zero_grad()
    #     loss = - self.embedding_LL(int_loc_optim[0])
    #     make_dot(loss, params={'leaf_loc': int_loc}).render('loss_torchviz', format='png')
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     scheduler.step()
    #     print('iteration %-12i Likelihood: %10.3f' % (iter*i, -loss))
    #     LL.append(loss)
    #     leaf_locs = self.VariationalParams["leaf_mu"]
    #     int_locs = self.VariationalParams["int_mu"]
    #     leaf_poin = torch.sigmoid(leaf_locs) * 2 - 1
    #     int_poin = torch.sigmoid(int_locs) * 2 - 1
    #     int_r, int_dir = utils.cart_to_dir(int_poin)
    #     leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)

    #     peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)

    #     fig, ax = plt.subplots(1, 1)
    #     X = torch.cat((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, self.D)))
    #     tree.plot_tree(ax, peel, X.detach().numpy())
    #     plt.show()
    #     # compare to grid search
    #     # print('Grid seach give maximum LL: %f' % max_LL)

    def learn_ML_brute(self, param_init=None, rounds=20):
        """Learn a Maximum Likelihood embedding.

        Args:
            param_init ([type], optional): [description]. Defaults to None.
            iter_per_param (int, optional): [description]. Defaults to 100.
            rounds (int, optional): [description]. Defaults to 3.
            path_write (str, optional): [description]. Defaults to './out'.
            lr ([type], optional): [description]. Defaults to 1e-3.
        """
        print("Learning ML tree.", flush=True)
        assert self.connector == "mst"
        LL = []

        if param_init is not None:
            self.VariationalParams["leaf_mu"] = param_init["leaf_mu"]
            self.VariationalParams["leaf_sigma"] = param_init["leaf_sigma"]
            if self.connector == "mst":
                self.VariationalParams["int_mu"] = param_init["int_mu"]
                self.VariationalParams["int_sigma"] = param_init["int_sigma"]

        iteration = 0
        for _ in range(rounds):
            isLeaf = False
            for i in range(self.S - 2):
                # optimise one internal node at a time
                with torch.no_grad():
                    max_LL = self.grid_search_LL(i, isLeaf=isLeaf, doPlot=True)
                LL.append(max_LL)
                print("iteration %-12i Likelihood: %10.3f" % (iteration, max_LL))
                iteration += 1

            # isLeaf = True
            # for i in range(self.S):
            #     # optimise one leaf node at a time
            #     with torch.no_grad():
            #         max_LL = self.grid_search_LL(i, isLeaf=isLeaf)
            #     LL.append(max_LL)
            #     print('iteration %-12i Likelihood: %10.3f' % (iter, max_LL))
            #     iter += 1

        # plot final tree
        leaf_locs = self.VariationalParams["leaf_mu"]
        int_locs = self.VariationalParams["int_mu"]
        if self.embedder == "simple":
            leaf_poin = utils.real2ball(leaf_locs)
            int_poin = utils.real2ball(int_locs)
        elif self.embedder == "wrap":
            leaf_poin = Chyp_torch.t02p(leaf_locs)
            int_poin = Chyp_torch.t02p(int_locs)
        int_r, int_dir = utils.cart_to_dir(int_poin)
        leaf_r, leaf_dir = utils.cart_to_dir(leaf_poin)
        peel = peeler.make_peel_mst(leaf_r, leaf_dir, int_r, int_dir)
        _, ax = plt.subplots(1, 1)
        X = torch.cat((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, self.D)))
        tree.plot_tree(ax, peel, X.detach().numpy())
        plt.show()

    def grid_search_LL(self, idx, isLeaf=True, doPlot=False):
        """Find Maximum likelihood tree by only move one internal node.

        Args:
            idx (_type_): _description_
            isLeaf (bool, optional): _description_. Defaults to False.
            doPlot (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert self.D == 2
        leaf_locs = self.param_init["leaf_loc"]
        if self.internals_exist:
            int_locs = self.param_init["int_loc"]

        # convert current nodes to poincare ?

        # beat the current likelihood
        if isLeaf:
            best_loc = leaf_poin[idx, :]
        else:
            best_loc = int_poin[idx, :]
        pdm = Chyp_np.get_pdm(leaf_locs, curvature=self.curvature)
        peel, blen = Cpeeler.nj_np(pdm)
        best_lnLike = self.compute_LL(peel, blen)

        # grid search centred at current location in Poincare disk
        # scale = torch.pow(torch.sum(torch.pow(best_loc, 2), axis=-1), .5)
        # X = torch.linspace(-scale, scale, steps) + best_loc[0].detach().numpy()
        # Y = torch.linspace(-scale, scale, steps) + best_loc[1].detach().numpy()

        # grid search over [-.5, .5]^2
        steps = 10
        X = torch.linspace(-0.5, 0.5, steps)
        Y = torch.linspace(-0.5, 0.5, steps)
        lnLike = torch.zeros((steps, steps))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                test_point = torch.tensor([x, y])
                # if torch.norm(test_point) > 1:
                #     lnLike[j, i] = -np.inf
                #     continue
                if isLeaf:
                    leaf_poin[idx, :] = test_point
                else:
                    int_poin[idx, :] = test_point
                
                pdm = Chyp_np.get_pdm(leaf_locs, curvature=self.curvature)
                peel, blen = Cpeeler.nj_np(pdm)
                cur_ll = self.compute_LL(peel, blen)
                lnLike[j, i] = cur_ll
                if cur_ll > best_lnLike:
                    best_lnLike = cur_ll
                    best_loc = test_point

        # contour plot of best positions
        if doPlot:
            _, ax = plt.subplots(1, 2)
            X, Y = np.meshgrid(X, Y)
            ax[0].contourf(X, Y, lnLike, cmap="hot", levels=200)
            ax[0].scatter(best_loc[:, 0], best_loc[:, 1])
            # plt.colorbar()
            if not isLeaf:
                ax[0].set_title("Node %i Likelihood" % (self.S + idx))
            else:
                ax[0].set_title("Node %i Likelihood" % idx)

            # plot best tree
            # if isLeaf:
            #     leaf_poin[idx, :] = best_loc
            # else:
            #     int_poin[idx, :] = best_loc
            # pdm = Chyp_np.get_pdm(leaf_locs, curvature=self.curvature)
            # peel, blen = Cpeeler.nj_np(pdm)
            # locs = torch.cat((leaf_poin, int_poin, leaf_poin[0, :].reshape(1, self.D)))
            # tree.plot_tree(ax[1], peel, locs)
            plt.show()

        if isLeaf:
            if self.embedder == "simple":
                self.VariationalParams["leaf_mu"][idx, :] = Chyperboloid.ball2real(
                    best_loc
                )
            elif self.embedder == "wrap":
                self.VariationalParams["leaf_mu"][idx, :] = Chyperboloid.p2t0(best_loc)
        else:
            if self.embedder == "simple":
                self.VariationalParams["int_mu"][idx, :] = Chyperboloid.ball2real(
                    best_loc
                )
            elif self.embedder == "wrap":
                self.VariationalParams["int_mu"][idx, :] = Chyperboloid.p2t0(best_loc)
        return best_lnLike
