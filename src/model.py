import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
import numpy as np
from tqdm import trange
from tqdm.notebook import tqdm

from utils import utilFunc


class DodonaphyModel(Distribution):
    data = {}
    VarationalParams = {
        "leaf_r_mu": torch.randn(1, requires_grad=True),
        "leaf_r_sigma": torch.randn(1, requires_grad=True),
        "leaf_dir_mu": torch.randn(1, requires_grad=True),
        "leaf_dir_sigma": torch.randn(1, requires_grad=True),
        "int_r_mu": torch.randn(1, requires_grad=True),
        "int_r_sigma": torch.randn(1, requires_grad=True),
        "int_dir_mu": torch.randn(1, requires_grad=True),
        "int_dir_sigma": torch.randn(1, requires_grad=True)
    }

    def __init__(self, D, L, S):
        self.parameters = {
            "int_r": torch.empty(S-2, requires_grad = True),        # radial distance
            "int_dir": torch.empty(S-2, D, requires_grad=True),  # angles
            "leaf_r": torch.empty(S, requires_grad=True),           # adial distance of each tip sequence in the embedding
            # directional coordinates of each tip sequence in the embedding
            "leaf_dir": torch.empty(S,D, requires_grad=True)
        }
        self.transformedData = {
            "bcount": 2*S-2
        }
        self.generatedQuantities = {
            "peel": np.zeros((S-1,3)),   # tree topology
            "blens": torch.empty(2*S-2, requires_grad = True)     # branch lengths
        }

    def compute_LL(self, S, L, bcount, D, tipdata, leaf_r, leaf_dir, int_r, int_dir):
        partials = torch.empty(2*S, L, 4, requires_grad=True)   # partial probabilities for the S tips and S-1 internal nodes
        fttm = torch.empty(bcount, 4, 4, requires_grad=True)    # finite-time transition matrices for each branch
        blens = torch.empty(bcount, requires_grad=True)         # brach lenghts
        peel = np.zeros(S-1,3)                                  # list of nodes for peeling.
        logprob = 0

        utilFunc.make_peel(leaf_r, leaf_dir, int_r, int_dir, peel, location_map)
        blens = self.compute_branch_lengths(S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir)

        # compute the finite time transition matrices
        for b in range(bcount):
            for i in range(4):
                for j in range(4):
                    fttm[b,i,j] = 0.25 - 0.25*torch.exp(-4*blens[b]/3)
                fttm[b,i,i] = 0.25 + 0.75*torch.exp(-4*blens[b]/3)
        # copy tip data into node probability vector
        for n in range(S):
            for i in range(L):
                for a in range(4):
                    partials[n,i,a] = tipdata[n,i,a]

        # calculate tree likelihood for the topoloty encoded in peel
        # for( i in 1:L ) {
		# for( n in 1:(S-1) ) {
		# 	partials[peel[n,3],i] = (fttm[peel[n,1]]*partials[peel[n,1],i]) .* (fttm[peel[n,2]]*partials[peel[n,2],i]);
		# }

		# // multiply by background nt freqs (assuming uniform here)
		# for(j in 1:4){
		# 	partials[2*S,i][j] = partials[peel[S-1,3],i][j] * 0.25;
		# }
		# // add the site log likelihood
		# logprob += log(sum(partials[2*S,i]));
	    # }
        for i in range(L):
            for n in range(S-1):
                partials[peel[n,3],i] = fttm[peel[n,1]]*partials[peel[n,1],i]  fttm[peel[n,2]]*partials[peel[n,2],i]]


    # computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
    # code taken from hydra R package and translated to pytorch
    def hyperbolic_distance(r1, r2, directional1, directional2, curvature):
        # r1 = torch.tensor(r1, requires_grad = True)
        # r2 = torch.tensor(r2, requires_grad = True)
        # directional1 = torch.from_numpy(directional1, requires_grad = True)
        # directional2 = torch.from_numpy(directional2, requires_grad = True)
        dpl = torch.empty(2)
        dpl[0] = torch.dot(directional1, directional2)
        dpl[1] = torch.tensor([1.0], requires_grad=True)
        iprod = dpl[0]
        dpl[0] = 2 * (torch.pow(r1, 2) + torch.pow(r2, 2) - 2*r1*r2*iprod)/((1-torch.pow(r1,2) * (1-torch.pow(r2,2))))
        dpl[1]= 0.0
        acosharg = 1.0 + max(dpl)
        # hyperbolic distance between points i and j
        dist = 1/math.sqrt(curvature) * torch.cosh(acosharg)
        return dist + 0.000000000001; # add a tiny amount to avoid zero-length branches

    
    def compute_branch_lengths(self, S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir):
        bcount = 2*S-2
        blens = torch.empty(bcount)
        for b in range(1, S-1):
            directional1, directional2 = torch.empty(D, requires_grad=True), torch.empty(D, requires_grad=True)
            directional2 = int_dir[location_map[peel[b,3]]-S, :]
            r1 = torch.empty(1)
            r2 = int_r[location_map[peel[b,3]]-S]
            if peel[b,1] <= S:
                # leaf to internal
                r1 = leaf_r[peel[b,1]]
                directional1 = leaf_dir[peel[b,1], :]
            elif:
                # internal to internal
                r1 = int_r[location_map[peel[b,1]]-S]
                directional1 = int_dir[location_map[peel[b,1]]-S, :]
            # apply the inverse transform from Matsumoto et al 2020
            blens[peel[b,1]] = torch.log(torch.cosh(blens[peel[b,1]])) + 0.000000000001; # add a tiny amount to avoid zero-length branches

            if peel[b,2] <= S:
                # leaf to internal
                r1 = leaf_r[peel[b,2]]
                directional1 = leaf_dir[peel[b,2], :]
            elif:
                # internal to internal
                r1 = int_r[location_map[peel[b,2]]-S]
                directional1 = int_dir[location_map[peel[b,2]-S],]
            blens[peel[b,2]] = hyperbolic_distance(r1, r2, directional1, directional2, 1)

            # apply the inverse transform from Matsumoto et al 2020
            blens[peel[b,2]] = torch.log(torch.cosh(blens[peel[b,2]])) +0.000000000001;  # add a tiny amount to avoid zero-length branches

        return blens


    def draw_sample(self):
        placeholder = 0

    # def log_p(self):
    #     placeholder = 0

    def calculate_elbo(self, q_leaf_r, q_leaf_dir, q_int_r, q_int_dir):
        # z
        z_leaf_r = q_leaf_r.rsample()
        z_leaf_dir = q_leaf_dir.rsample()
        z_int_r = q_int_r.rsample()
        z_int_dir = q_int_dir.rsample()
        # logQ
        logQ_leaf_r = q_leaf_r.log_prob(z_leaf_r)
        logQ_leaf_dir = q_leaf_dir.log_prob(z_leaf_dir)
        logQ_int_r = q_int_r.log_prob(z_int_r)
        logQ_int_dir = q_int_dir.log_prob(z_int_dir)
        logQ = logQ_leaf_r + logQ_leaf_dir + logQ_int_r + logQ_int_dir

        # logPrior, have to think carefully

        # logP
        log_P_leaf_r =

        return logP + logPrior - logQ   

    def learn(self, dpy_dat, param_init, epoch=1000):
        def lr_lambda(epoch): return 1.0/np.sqrt(epoch+1)

        # set data
        self.data = dpy_dat

        # set initial params
        VarationalParams["leaf_r_mu"] = param_init["leaf_r"].mean()
        VarationalParams["leaf_r_sigma"] = param_init["leaf_r"].std()
        VarationalParams["leaf_dir_mu"] = param_init["leaf_dir"].mean()
        VarationalParams["leaf_dir_sigma"] = param_init["leaf_dir"].std()
        VarationalParams["int_r_mu"] = param_init["int_r"].mean()
        VarationalParams["int_r_sigma"] = param_init["int_r"].std()
        VarationalParams["int_dir_mu"] = param_init["int_dir"].mean()
        VarationalParams["int_dir_sigma"] = param_init["int_dir"].std()

        dodonaphy_mod = self

        optimizer = torch.optim.Adam(VarationalParams, lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

        elbo_hist = []
        mu_grads = {}
        sigma_grads = {}
        mus = {}
        sigmas = {}

        iters = tqdm(range(50000), mininterval=1)
        for epoch in iters:
            mus.append(theta_mu.item())
            sigmas.append(theta_sigma.exp().item())
            
            loss = -elbo_lognormal()
            
            elbo_hist.append(-loss.item())
            optimizer.zero_grad()
            loss.backward()
            mu_grads.append(theta_mu.grad.item())
            sigma_grads.append(theta_sigma.grad.item()**2)
            optimizer.step()
            scheduler.step()
            iters.set_description('ELBO: {}'.format(elbo_hist[-1]), refresh=False)
        
        with torch.no_grad():
            print('Final ELBO: {}'.format(elbo_lognormal(100).item()))

    def elbo_normal(self, var_params, size=1):
        elbo = 0
        # q_thetas
        q_leaf_r = torch.distributions.LogNormal(
            var_params.leaf_r_mu, var_params.leaf_r_sigma.exp())
        q_leaf_dir = torch.distributions.LogNormal(
            var_params.leaf_dir_mu, var_params.leaf_dir_sigma.exp())
        q_int_r = torch.distributions.LogNormal(
            var_params.int_r_mu, var_params.int_r_sigma.exp())
        q_int_dir = torch.distributions.LogNormal(
            var_params.int_dir_mu, var_params.int_dir_sigma.exp())
        for i in range(size):
            elbo += self.calculate_elbo(q_leaf_r,
                                        q_leaf_dir, q_int_r, q_int_dir)
        return elbo/size


