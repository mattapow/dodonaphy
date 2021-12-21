# from matplotlib.markers import MarkerStyle
# from dodonaphy import poincare
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dodonaphy import Chyperboloid, utils
from matplotlib.patches import Circle
from torch.distributions.multivariate_normal import MultivariateNormal

# sampling parameters
loc_og = torch.tensor([0.00001, .000001]).double()
cov = torch.eye(2).double()
zero = torch.zeros_like(loc_og)
n_sample = 10000
sample_shape = torch.Size([n_sample])
_, ax = plt.subplots(1, 2, sharey=True, sharex=True)

# simple
loc = Chyperboloid.ball2real(loc_og, radius=2.).squeeze()
normal_dist = MultivariateNormal(loc, cov)
sample = normal_dist.sample(sample_shape)
loc_poin_simple = Chyperboloid.real2ball(sample, radius=2.)
sns.kdeplot(ax=ax[0], x=loc_poin_simple[:, 0], y=loc_poin_simple[:, 1],
            fill=True, levels=100)
# ax[0].plot(loc_poin_simple[:, 0], loc_poin_simple[:, 1], 'r.', ms=.3)
ax[0].set_title("Simple")

# wrap
loc = Chyperboloid.p2t0(loc_og)
normal_dist = MultivariateNormal(zero, cov)
sample = normal_dist.sample(sample_shape)
loc_poin_wrap = Chyperboloid.t02p(sample, mu=loc.T.repeat((n_sample, 1)))
sns.kdeplot(ax=ax[1], x=loc_poin_wrap[:, 0], y=loc_poin_wrap[:, 1],
            fill=True, levels=100)
# ax[1].plot(loc_poin_wrap[:, 0], loc_poin_wrap[:, 1], 'r.', ms=.3)
ax[1].set_title("Wrap")

# # exp
# dir = loc_og / torch.norm(loc_og, dim=-1, keepdim=True)
# sigma = 1.
# loc = dir * sigma * (-torch.log((1-torch.norm(loc_og, dim=-1))))**.5
# normal_dist = MultivariateNormal(loc, cov / torch.norm(loc)**2)
# sample = normal_dist.sample(sample_shape)
# dir_sample = sample / torch.norm(sample, dim=1, keepdim=True)
# loc_poin_exp = dir_sample * (1. - torch.exp(-torch.norm(sample, dim=1, keepdim=True)**2 / sigma**2))
# sns.kdeplot(ax=ax[2], x=loc_poin_exp[:, 0], y=loc_poin_exp[:, 1],
#             fill=True, levels=100)
# # ax[2].plot(loc_poin_exp[:, 0], loc_poin_exp[:, 1], 'r.', ms=.3)
# ax[2].set_title("Exp")


circ = Circle((0, 0), radius=1., fill=False, edgecolor='k')
ax[0].add_patch(circ)
circ = Circle((0, 0), radius=1., fill=False, edgecolor='k')
ax[1].add_patch(circ)
# circ = Circle((0, 0), radius=1., fill=False, edgecolor='k')
# ax[2].add_patch(circ)
plt.show()
