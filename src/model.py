import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
import numpy as np
from tqdm import trange
from tqdm.notebook import tqdm
from utils import utilFunc
from torch.distributions import SigmoidTransform


class DodonaphyModel(Distribution):
    data = {}
    # VarationalParams = {
    #     "leaf_r_mu": torch.randn(1, requires_grad=True),
    #     "leaf_r_sigma": torch.randn(1, requires_grad=True),
    #     "leaf_dir_mu": torch.randn(1, requires_grad=True),
    #     "leaf_dir_sigma": torch.randn(1, requires_grad=True),
    #     "int_r_mu": torch.randn(1, requires_grad=True),
    #     "int_r_sigma": torch.randn(1, requires_grad=True),
    #     "int_dir_mu": torch.randn(1, requires_grad=True),
    #     "int_dir_sigma": torch.randn(1, requires_grad=True)
    # }

    def __init__(self, D, L, S):
        # self.parameters = {
        #     # radial distance
        #     "int_r": torch.empty(S-2, requires_grad=True),
        #     "int_dir": torch.empty(S-2, D, requires_grad=True),  # angles
        #     # adial distance of each tip sequence in the embedding
        #     "leaf_r": torch.empty(S, requires_grad=True),
        #     # directional coordinates of each tip sequence in the embedding
        #     "leaf_dir": torch.empty(S, D, requires_grad=True)
        # }
        self.VarationalParams = {
            "leaf_r_mu": torch.randn(1, requires_grad=True),
            "leaf_r_sigma": torch.randn(1, requires_grad=True),
            "leaf_dir_mu": torch.randn(1, requires_grad=True),
            "leaf_dir_sigma": torch.randn(1, requires_grad=True),
            "int_r_mu": torch.randn(1, requires_grad=True),
            "int_r_sigma": torch.randn(1, requires_grad=True),
            "int_dir_mu": torch.randn(1, requires_grad=True),
            "int_dir_sigma": torch.randn(1, requires_grad=True)
        }
        self.transformedData = {
            "bcount": 2*S-2
        }
        self.generatedQuantities = {
            "peel": np.zeros((S-1, 3)),   # tree topology
            # branch lengths
            "blens": torch.empty(2*S-2, requires_grad=True)
        }


    def compute_branch_lengths(self, S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir):
        """Computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
        
        Args:
            S (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            location_map ([type]): [description]
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        bcount = 2*S-2
        blens = torch.empty(bcount)
        for b in range(1, S-1):
            directional1, directional2 = torch.empty(
                D, requires_grad=True), torch.empty(D, requires_grad=True)
            directional2 = int_dir[location_map[peel[b, 3]]-S, :]
            r1 = torch.empty(1)
            r2 = int_r[location_map[peel[b, 3]]-S]
            if peel[b, 1] <= S:
                # leaf to internal
                r1 = leaf_r[peel[b, 1]]
                directional1 = leaf_dir[peel[b, 1], :]
            else:
                # internal to internal
                r1 = int_r[location_map[peel[b, 1]]-S]
                directional1 = int_dir[location_map[peel[b, 1]]-S, :]
            # apply the inverse transform from Matsumoto et al 2020
            # add a tiny amount to avoid zero-length branches
            blens[peel[b, 1]] = torch.log(
                torch.cosh(blens[peel[b, 1]])) + 0.000000000001

            if peel[b, 2] <= S:
                # leaf to internal
                r1 = leaf_r[peel[b, 2]]
                directional1 = leaf_dir[peel[b, 2], :]
            else:
                # internal to internal
                r1 = int_r[location_map[peel[b, 2]]-S]
                directional1 = int_dir[location_map[peel[b, 2]-S], ]
            blens[peel[b, 2]] = utilFunc.hyperbolic_distance(
                r1, r2, directional1, directional2, 1)

            # apply the inverse transform from Matsumoto et al 2020
            # add a tiny amount to avoid zero-length branches
            blens[peel[b, 2]] = torch.log(
                torch.cosh(blens[peel[b, 2]])) + 0.000000000001

        return blens

    def compute_LL(self, leaf_r, leaf_dir, int_r, int_dir):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """
        S = self.data["S"]
        L = self.data["L"]
        D = self.data["D"]
        tipdata = self.data["tipdata"]
        bcount = self.transformedData["bcount"]

        # partial probabilities for the S tips and S-1 internal nodes
        partials = torch.empty(2*S, L, 4, requires_grad=True)
        # finite-time transition matrices for each branch
        fttm = torch.empty(bcount, 4, 4, requires_grad=True)
        # list of nodes for peeling.
        peel = np.zeros(S-1, 3)
        # node location map
        location_map = np.empty(2*S-1, dtype=np.int64)

        utilFunc.make_peel(leaf_r, leaf_dir, int_r,
                           int_dir, peel, location_map)
        # brach lenghts
        blens = self.compute_branch_lengths(
            S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir)

        # # compute the finite time transition matrices
        # for b in range(bcount):
        #     for i in range(4):
        #         for j in range(4):
        #             fttm[b,i,j] = 0.25 - 0.25*torch.exp(-4*blens[b]/3)
        #         fttm[b,i,i] = 0.25 + 0.75*torch.exp(-4*blens[b]/3)
        # # copy tip data into node probability vector
        # for n in range(S):
        #     for i in range(L):
        #         for a in range(4):
        #             partials[n,i,a] = tipdata[n,i,a]

        # # calculate tree likelihood for the topoloty encoded in peel
        # # for( i in 1:L ) {
        # # for( n in 1:(S-1) ) {
        # # 	partials[peel[n,3],i] = (fttm[peel[n,1]]*partials[peel[n,1],i]) .* (fttm[peel[n,2]]*partials[peel[n,2],i]);
        # # }

        # # // multiply by background nt freqs (assuming uniform here)
        # # for(j in 1:4){
        # # 	partials[2*S,i][j] = partials[peel[S-1,3],i][j] * 0.25;
        # # }
        # # // add the site log likelihood
        # # logprob += log(sum(partials[2*S,i]));
        # # }
        # for i in range(L):
        #     for n in range(S-1):
        #         partials[peel[n,3],i] = fttm[peel[n,1]]*partials[peel[n,1],i]  fttm[peel[n,2]]*partials[peel[n,2],i]]

        return 0

    def draw_sample(self):
        """[summary]
        """
        placeholder = 0

    def calculate_elbo(self, q_leaf_r, q_leaf_dir, q_int_r, q_int_dir):
        """[summary]

        Args:
            q_leaf_r ([type]): [description]
            q_leaf_dir ([type]): [description]
            q_int_r ([type]): [description]
            q_int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        sigmoid_transformation = SigmoidTransform()

        
        # z
        z_leaf_r = q_leaf_r.rsample()
        z_leaf_dir = q_leaf_dir.rsample()
        z_int_r = q_int_r.rsample()
        z_int_dir = q_int_dir.rsample()
        # transformation of z
        z_leaf_r = sigmoid_transformation(z_leaf_r)        
        z_leaf_dir = sigmoid_transformation((z_leaf_dir - 1) / 2)
        z_int_r = sigmoid_transformation(z_int_r)
        z_int_dir = sigmoid_transformation((z_int_dir - 1) / 2)


        # logQ
        logQ_leaf_r = q_leaf_r.log_prob(z_leaf_r)
        logQ_leaf_dir = q_leaf_dir.log_prob(z_leaf_dir)
        logQ_int_r = q_int_r.log_prob(z_int_r)
        logQ_int_dir = q_int_dir.log_prob(z_int_dir)
        logQ = logQ_leaf_r + logQ_leaf_dir + logQ_int_r + logQ_int_dir

        # logPrior, have to think carefully
        logPrior = 0

        # logP
        logP = self.compute_LL(z_leaf_r, z_leaf_dir, z_int_r, z_int_dir)

        return logP + logPrior - logQ

    def learn(self, dpy_dat, param_init, epoch=1000):
        """[summary]

        Args:
            dpy_dat ([type]): [description]
            param_init ([type]): [description]
            epoch (int, optional): [description]. Defaults to 1000.
        """
        def lr_lambda(epoch): return 1.0/np.sqrt(epoch+1)

        # set data
        # dpy_dat is a dict (keys are: S, L, D, tipdata)
        self.data = dpy_dat

        # set initial params as a Dict
        self.VarationalParams["leaf_r_mu"], self.VarationalParams["leaf_r_sigma"] = torch.tensor(param_init["leaf_r"].mean(
        ), requires_grad=True), torch.tensor(param_init["leaf_r"].std(), requires_grad=True)

        self.VarationalParams["leaf_dir_mu"], self.VarationalParams["leaf_dir_sigma"] = torch.tensor(param_init["leaf_dir"].mean(
        ), requires_grad=True), torch.tensor(param_init["leaf_dir"].std(), requires_grad=True)
        self.VarationalParams["int_r_mu"], self.VarationalParams["int_r_sigma"] = torch.tensor(param_init["int_r"].mean(
        ), requires_grad=True), torch.tensor(param_init["int_r"].std(), requires_grad=True)
        self.VarationalParams["int_dir_mu"], self.VarationalParams["int_dir_sigma"] = torch.tensor(param_init["int_dir"].mean(
        ), requires_grad=True), torch.tensor(param_init["int_dir"].std(), requires_grad=True)

        dodonaphy_mod = self

        optimizer = torch.optim.Adam(list(self.VarationalParams.values()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        mu_grads = {
            'leaf_r_mu': [],
            'leaf_dir_mu': [],
            'int_r_mu': [],
            'int_dir_mu': []
        }
        sigma_grads = {
            'leaf_r_sigma': [],
            'leaf_dir_sigma': [],
            'int_r_sigma': [],
            'int_dir_sigma': []
        }
        mus = {
            'leaf_r_mu': [],
            'leaf_dir_mu': [],
            'int_r_mu': [],
            'int_dir_mu': []
        }
        sigmas = {
            'leaf_r_sigma': [],
            'leaf_dir_sigma': [],
            'int_r_sigma': [],
            'int_dir_sigma': []
        }

        for epoch in range(50000):
            # save mus
            mus['leaf_r_mu'].append(self.VarationalParams["leaf_r_mu"].item())
            mus['leaf_dir_mu'].append(
                self.VarationalParams["leaf_dir_mu"].item())
            mus['int_r_mu'].append(self.VarationalParams["int_r_mu"].item())
            mus['int_dir_mu'].append(
                self.VarationalParams["int_dir_mu"].item())
            # save sigmas
            sigmas['leaf_r_sigma'].append(
                self.VarationalParams["leaf_r_sigma"].exp().item())
            sigmas['leaf_dir_sigma'].append(
                self.VarationalParams["leaf_dir_sigma"].exp().item())
            sigmas['int_r_sigma'].append(
                self.VarationalParams["int_r_sigma"].exp().item())
            sigmas['int_dir_sigma'].append(
                self.VarationalParams["int_dir_sigma"].exp().item())

            loss = - self.elbo_normal()

            elbo_hist.append(-loss.item())
            optimizer.zero_grad()
            loss.backward()
            # save mu grads
            mu_grads['leaf_r_mu'].append(
                self.VarationalParams["leaf_r_mu"].grad.item())
            mu_grads['leaf_dir_mu'].append(
                self.VarationalParams["leaf_dir_mu"].grad.item())
            mu_grads['int_r_mu'].append(
                self.VarationalParams["int_r_mu"].grad.item())
            mu_grads['int_dir_mu'].append(
                self.VarationalParams["int_dir_mu"].grad.item())
            # save sigma grads
            sigma_grads['leaf_r_sigma'].append(
                self.VarationalParams["leaf_r_sigma"].exp().item()**2)
            sigma_grads['leaf_dir_sigma'].append(
                self.VarationalParams["leaf_dir_sigma"].exp().item()**2)
            sigma_grads['int_r_sigma'].append(
                self.VarationalParams["int_r_sigma"].exp().item()**2)
            sigma_grads['int_dir_sigma'].append(
                self.VarationalParams["int_dir_sigma"].exp().item()**2)
            optimizer.step()
            scheduler.step()
            'ELBO: {}'.format(elbo_hist[-1])

        with torch.no_grad():
            print('Final ELBO: {}'.format(self.elbo_normal(100).item()))

    def elbo_normal(self, size=1):
        """[summary]

        Args:
            size (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        elbo = 0
        # q_thetas
        q_leaf_r = torch.distributions.Normal(
            self.VarationalParams["leaf_r_mu"].item(), 
            math.exp(self.VarationalParams["leaf_r_sigma"].item()))
        q_leaf_dir = torch.distributions.Normal(
            self.VarationalParams["leaf_dir_mu"].item(), 
            math.exp(self.VarationalParams["leaf_dir_sigma"].item()))
        q_int_r = torch.distributions.Normal(
            self.VarationalParams["int_r_mu"].item(), 
            math.exp(self.VarationalParams["int_r_sigma"].item()))
        q_int_dir = torch.distributions.Normal(
            self.VarationalParams["int_dir_mu"].item(), 
            math.exp(self.VarationalParams["int_dir_sigma"].item()))

        for i in range(size):
            elbo += self.calculate_elbo(q_leaf_r,
                                        q_leaf_dir, q_int_r, q_int_dir)
        return elbo/size

# mymod = DodonaphyModel(3,1000,3)
# print(mymod.data)
