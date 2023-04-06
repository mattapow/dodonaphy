import torch


class PhyloModel():
    def __init__(self, name):
        self.name = name
        self.init_freqs()
        self.init_sub_rates()
        self.init_fix_params()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

    @property
    def freqs(self):
        freq4 = 1.0 - torch.sum(self._freqs, dim=0, keepdim=True)
        return torch.cat((self._freqs, freq4))

    @freqs.setter
    def freqs(self, freqs):
        self._freqs = freqs[:3]

    def init_freqs(self):
        self.freqs = torch.full([4], 0.25, dtype=torch.double)
        self.freqs_prior_dist = torch.distributions.dirichlet.Dirichlet(
            torch.tensor([1.0, 1.0, 1.0, 1.0])
        )

    def init_sub_rates(self):
        if self == "JC69":
            self.sub_rates = torch.empty([])
        elif self == "GTR":
            self.prior_dist = torch.distributions.dirichlet.Dirichlet(
                torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            )
            self.sub_rates = self.prior_dist.sample()
        else:
            raise RuntimeError("Model not implemented")

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
        return torch.cat(
            (a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
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
            (sqrt_pi_inv @ v).reshape(e.shape[:-1] + (1,) * offset + sqrt_pi_inv.shape[-2:])
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

    # TODO: save the model. Be careful with VI and HMAP where we optimise the
    # variational parameters, not the parameters saved here.
    def print_model():
        return None

    def compute_ln_prior_sub_rates(self, sub_rates):
        if self == "JC69":
            return torch.zeros(1)
        elif self == "GTR":
            return self.prior_dist.log_prob(sub_rates)
        else:
            raise RuntimeError(f"Model {self.name} has no prior on rates available.")

    def compute_ln_prior_freqs(self, freqs):
        # TODO work in simplex. For now, just normalise freqs
        freqs = freqs / sum(freqs)
        return self.freqs_prior_dist.log_prob(freqs)