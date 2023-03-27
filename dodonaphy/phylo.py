from collections import Counter

import numpy as np
import torch


def compress_alignment(alignment, get_namespace=False):
    sequences = [str(sequence) for sequence in alignment.sequences()]
    taxa = alignment.taxon_namespace.labels()
    count_dict = Counter(list(zip(*sequences)))
    pattern_ordering = sorted(list(count_dict.keys()))
    patterns_list = list(zip(*pattern_ordering))
    weights = [count_dict[pattern] for pattern in pattern_ordering]
    patterns = dict(zip(taxa, patterns_list))

    partials = []

    dna_map = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "C": [0.0, 1.0, 0.0, 0.0],
        "G": [0.0, 0.0, 1.0, 0.0],
        "T": [0.0, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 1.0, 0.0],
        "Y": [0.0, 1.0, 0.0, 1.0],
        "M": [1.0, 1.0, 0.0, 0.0],
        "W": [1.0, 0.0, 0.0, 1.0],
        "S": [0.0, 1.0, 1.0, 0.0],
        "K": [0.0, 0.0, 1.0, 1.0],
        "B": [0.0, 1.0, 1.0, 1.0],
        "D": [1.0, 0.0, 1.0, 1.0],
        "H": [1.0, 1.0, 0.0, 1.0],
        "V": [1.0, 1.0, 1.0, 0.0],
        "N": [1.0, 1.0, 1.0, 1.0],
        "?": [1.0, 1.0, 1.0, 1.0],
        "-": [1.0, 1.0, 1.0, 1.0],
    }
    unknown = [1.0] * 4

    for taxon in taxa:
        partials.append(
            torch.tensor(
                np.transpose(
                    np.array([dna_map.get(c.upper(), unknown) for c in patterns[taxon]])
                )
            )
        )
    if get_namespace:
        return partials, torch.tensor(np.array(weights)), alignment.taxon_namespace
    return partials, torch.tensor(np.array(weights))


def calculate_pairwise_distance(dna, adjust=None, omit_unkown=False):
    """Calculate the pairwise evolutionary distances.

    The evolutionary distance rho is the number of substitutions per site.
    These distance can be corrected under a JC69 model:
        d = - 3/4 ln(1 - 4/3 rho).
    Treats gaps as no distance, but does count towards the total alignment length.

    Args:
        dna (DnaCharacterMatrix): From Dendropy.
        adjust (String, optional): Set to 'JC69' to adjust for JC69 distances.
        Defaults to None.
        omit_unkown (Boolean, optional): Reduce total length if either sequence
        has unknwon. Defaults to False.

    Returns:
        array: A 2d array of the pairwise evolutionary distances.
    """
    n_taxa = len(dna)
    seq_len = len(dna[0])
    obs_dist = np.zeros((n_taxa, n_taxa))
    count_unknown = np.zeros((n_taxa, n_taxa))
    for i in range(n_taxa):
        seq_i = dna[i]
        for j in range(i + 1, n_taxa):
            seq_j = dna[j]
            # count the number of different characters, excluding any gaps
            count = sum(
                seq_i[k] != seq_j[k]
                and not seq_i[k].is_gap_state
                and not seq_j[k].is_gap_state
                for k in range(seq_len)
            )
            obs_dist[i, j] += count
            obs_dist[j, i] += count
            # count how many gaps are in either sequence
            count_unknown[i, j] = sum(
                seq_i[k].is_gap_state or seq_j[k].is_gap_state for k in range(seq_len)
            )
            count_unknown[j, i] = count_unknown[i, j]

    # adjust the length of the alignment for unkown characters, for each pair of sequences
    if omit_unkown:
        seq_len = seq_len - count_unknown
        np.fill_diagonal(seq_len, 0)
    obs_dist /= seq_len
    np.fill_diagonal(obs_dist, 0)

    if adjust is None:
        return obs_dist
    elif adjust == "JC69":
        return -3 / 4 * np.log(1.0 - 4 / 3 * obs_dist)
    else:
        raise ValueError("adjust must be None or 'JC69'")


def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs):
    for left, right, node in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(
            mats[right], partials[right]
        )
    return torch.sum(
        torch.log(torch.matmul(freqs, partials[post_indexing[-1][-1]])) * weights
    )


def JC69_p_t(branch_lengths):
    d = torch.unsqueeze(branch_lengths, -1)
    a = 0.25 + 3.0 / 4.0 * torch.exp(-4.0 / 3.0 * d)
    b = 0.25 - 0.25 * torch.exp(-4.0 / 3.0 * d)
    return torch.cat((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
        d.shape[0], d.shape[1], 4, 4
    )


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


def norm(Q, freqs) -> torch.Tensor:
    return -torch.sum(torch.diagonal(Q, dim1=-2, dim2=-1) * freqs, dim=-1)


def GTR_p_t(branch_lengths, sub_rates, freqs):
    Q_unnorm = compute_Q_martix(sub_rates)
    Q = Q_unnorm / norm(Q_unnorm, freqs).unsqueeze(-1).unsqueeze(-1)
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
