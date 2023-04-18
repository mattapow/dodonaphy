import torch
from torch.distributions.transforms import StickBreakingTransform
from torch.distributions.dirichlet import Dirichlet


class PhyloModel:
    def __init__(self, name):
        self.name = name
        self.init_freqs(torch.full([4], 0.25, dtype=torch.double))
        self.init_sub_rates(torch.full([6], 1.0/6.0, dtype=torch.double))
        self.init_fix_params()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

    @property
    def freqs(self):
        print(self._freqs)
        return self.freqs_transform(self._freqs)

    @freqs.setter
    def freqs(self, freqs):
        if not hasattr(self, "_freqs"):
            self._freqs = self.freqs_transform.inv(freqs).clone().detach().requires_grad_(True)
        else:
            self._freqs = self.freqs_transform.inv(freqs)

    @property
    def sub_rates(self):
        return self.sub_rates_transform(self._sub_rates)

    @sub_rates.setter
    def sub_rates(self, sub_rates):
        if not hasattr(self, "_sub_rates"):
            self._sub_rates = self.sub_rates_transform.inv(sub_rates).clone().detach().requires_grad_(True)
        else:
            self._sub_rates = self.sub_rates_transform.inv(sub_rates)

    def init_freqs(self, freqs):
        self.freqs_transform = StickBreakingTransform()
        self.freqs = freqs
        if self == "JC69":
            self.freqs_prior_dist = None
        elif self == "GTR":
            self.freqs_prior_dist = Dirichlet(torch.full([4], 0.25))

    def init_sub_rates(self, sub_rates):
        self.sub_rates_transform = StickBreakingTransform()
        self.sub_rates = sub_rates
        if self == "JC69":
            self.prior_dist = None
        elif self == "GTR":
            self.prior_dist = Dirichlet(torch.full([6], 1.0/6.0))

    def init_fix_params(self):
        self.fix_sub_rates = True
        self.fix_freqs = True
        if self == "GTR":
            self.fix_sub_rates = False
            self.fix_freqs = False

    def get_transition_mats(self, blens, sub_rates, freqs):
        if self == "JC69":
            mats = self.JC69_p_t(blens)
        elif self == "GTR":
            mats = self.GTR_p_t(blens, sub_rates, freqs)
        else:
            raise ValueError(
                f"Model {self.name} has no transition matrix implementation."
            )
        return mats

    @staticmethod
    def JC69_p_t(branch_lengths):
        d = torch.unsqueeze(branch_lengths, -1)
        a = 0.25 + 3.0 / 4.0 * torch.exp(-4.0 / 3.0 * d)
        b = 0.25 - 0.25 * torch.exp(-4.0 / 3.0 * d)
        return torch.cat((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
            d.shape[0], d.shape[1], 4, 4
        )

    @staticmethod
    def GTR_p_t(branch_lengths, sub_rates, freqs):
        Q_unnorm = PhyloModel.compute_Q_martix(sub_rates)
        Q = Q_unnorm / PhyloModel.norm(Q_unnorm, freqs).unsqueeze(-1).unsqueeze(-1)
        sqrt_pi = freqs.sqrt().diag_embed(dim1=-2, dim2=-1)
        sqrt_pi_inv = (1.0 / freqs.sqrt()).diag_embed(dim1=-2, dim2=-1)
        sqrt_pi = freqs.sqrt().diag_embed(dim1=-2, dim2=-1)
        sqrt_pi_inv = (1.0 / freqs.sqrt()).diag_embed(dim1=-2, dim2=-1)
        S = sqrt_pi @ Q @ sqrt_pi_inv
        e, v = torch.linalg.eigh(S)
        offset = branch_lengths.dim() - e.dim() + 1
        return (
            (sqrt_pi_inv @ v).reshape(
                e.shape[:-1] + (1,) * offset + sqrt_pi_inv.shape[-2:]
            )
            @ torch.exp(
                e.reshape(e.shape[:-1] + (1,) * offset + e.shape[-1:])
                * branch_lengths.unsqueeze(-1)
            ).diag_embed()
            @ (v.inverse() @ sqrt_pi).reshape(
                e.shape[:-1] + (1,) * offset + sqrt_pi_inv.shape[-2:]
            )
        )

    @staticmethod
    def norm(Q, freqs) -> torch.Tensor:
        return -torch.sum(torch.diagonal(Q, dim1=-2, dim2=-1) * freqs, dim=-1)

    @staticmethod
    def compute_Q_martix(sub_rates):
        (a, b, c, d, e, f) = sub_rates
        Q = torch.tensor(
            [
                [0, a, b, c],
                [0, 0, d, e],
                [0, 0, 0, f],
                [0, 0, 0, 0],
            ]
        )
        Q = Q + Q.t()
        diag_sum = -torch.sum(Q, dim=0)
        Q[range(4), range(4)] = diag_sum
        return Q

    def save(self, file_name):
        with open(file_name, "w") as f:
            f.write(f"Model: {self.name}\n")
            f.write(f"Frequencies: {self.freqs.detach().numpy()}\n")
            f.write(f"Frequencies Fixed: {self.fix_freqs}\n")
            f.write(f"Frequencies Prior: {str(self.freqs_prior_dist)}\n")
            f.write(f"Substitution Rates: {self.sub_rates.detach().numpy()}\n")
            f.write(f"Substitution Rates Fixed: {self.fix_sub_rates}\n")
            f.write(f"Substitution Rates Prior: {str(self.prior_dist)}\n")

    def compute_ln_prior_sub_rates(self, sub_rates):
        if self == "JC69":
            return torch.zeros(1)
        elif self == "GTR":
            return self.prior_dist.log_prob(sub_rates)
        else:
            raise RuntimeError(f"Model {self.name} has no prior on rates available.")

    def compute_ln_prior_freqs(self, freqs):
        if self == "JC69":
            return torch.zeros(1)
        elif self == "GTR":
            return self.freqs_prior_dist.log_prob(freqs)
        else:
            raise RuntimeError(f"Model {self.name} has no prior on freqs available.")
