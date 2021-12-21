from dodonaphy import vi, utils, peeler, tree
import matplotlib.pyplot as plt
import numpy as np
import torch

path_read = '/Users/151569/OneDrive - UTS/Projects/Dodonaphy/dodonaphy/data/T17_hypGEO/vi/simple_geodesics/d5_lr1_k1/vi_params/vi_10.csv'
VariationalParams = vi.read(path_read, internals=True)
dim = 4
taxa = 17

leaf_locs = torch.tensor(VariationalParams['leaf_mu']).view((taxa, dim))
peel, int_locs = peeler.make_peel_geodesic(leaf_locs)
leaf_r, leaf_dir = utils.cart_to_dir(leaf_locs)
int_r, int_dir = utils.cart_to_dir(int_locs)

locs = np.concatenate((leaf_locs, int_locs, leaf_locs[0, :].reshape(1, dim)))
tree.plot_tree(plt.gca(), peel, locs, color=(0, 0, 0), labels=True, radius=float(leaf_r[0]))
plt.show()