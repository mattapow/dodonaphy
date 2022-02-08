from collections import Counter

import numpy as np
import torch


def compress_alignment(alignment):
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
            np.transpose(
                np.array([dna_map.get(c.upper(), unknown) for c in patterns[taxon]])
            )
        )
    return partials, np.array(weights)


def calculate_pairwise_distance(dna):
    dna_np = np.array(dna.sequences())
    return (dna_np[:, None] != dna_np).sum(axis=2) / len(dna[0])


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
