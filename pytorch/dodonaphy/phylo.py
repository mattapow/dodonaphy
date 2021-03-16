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

    dna_map = {'A': [1.0, 0.0, 0.0, 0.0],
               'C': [0.0, 1.0, 0.0, 0.0],
               'G': [0.0, 0.0, 1.0, 0.0],
               'T': [0.0, 0.0, 0.0, 1.0]}
    unknown = [1.0] * 4

    for taxon in taxa:
        partials.append(
            torch.tensor(np.transpose(np.array([dna_map.get(c.upper(), unknown) for c in patterns[taxon]]))))
    return partials, torch.tensor(np.array(weights))


def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs):
    for node, left, right in post_indexing:
        partials[node] = torch.matmul(mats[left], partials[left]) * torch.matmul(mats[right], partials[right])
    return torch.sum(torch.log(torch.matmul(freqs, partials[post_indexing[-1][0]])) * weights)

def JC69_p_t(d):
    evec = np.array([
        1.0, 2.0, 0.0, 0.5, 1.0, -2.0, 0.5, 0.0, 1.0, 2.0, 0.0, -0.5, 1.0, -2.0, -0.5, 0.0
    ])

    ivec = np.array([
        0.25, 0.25, 0.25, 0.25, 0.125, -0.125, 0.125, -0.125, 0.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, -1.0, 0.0
    ])

    evalues = np.array(
        [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])
    return evec.reshape((4, 4)) @ (np.expand_dims(np.exp(evalues*d), axis=-1)*np.eye(4)) @ ivec.reshape((4, 4))