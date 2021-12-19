import math
import numpy as np
cimport numpy as np
from .Chyperboloid_np import hyperbolic_distance
from collections import Counter

def compress_alignment_np(alignment):
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
    return partials, np.array(weights, dtype=int)


def compute_LL_np(partials,
    np.ndarray[np.int_t, ndim=1] weights,
    np.ndarray[np.int_t, ndim=2] peel,
    np.ndarray[np.double_t, ndim=1] blen):
        """[summary]

        Args:
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]
        """
        mats = JC69_p_t(blen)
        return calculate_treelikelihood(
            partials,
            weights,
            peel,
            mats,
            np.full([4], 0.25, dtype=np.float64),
        )

cpdef calculate_treelikelihood(
    partials,
    np.ndarray[np.int_t, ndim=1] weights,
    np.ndarray[np.int_t, ndim=2] post_indexing,
    np.ndarray[np.double_t, ndim=4] mats,
    np.ndarray[np.double_t, ndim=1] freqs):

    cdef np.int_t left
    cdef np.int_t right
    cdef np.int_t node

    for left, right, node in post_indexing:
        partials[node] = np.matmul(mats[left], partials[left]) * np.matmul(
            mats[right], partials[right]
        )
    return np.sum(
        np.log(np.matmul(freqs, partials[post_indexing[-1][-1]])) * weights
    )


cpdef JC69_p_t(np.ndarray[np.double_t, ndim=1] branch_lengths):
    d = np.expand_dims(branch_lengths, axis=1)
    a = 0.25 + 3.0 / 4.0 * np.exp(-4.0 / 3.0 * d)
    b = 0.25 - 0.25 * np.exp(-4.0 / 3.0 * d)
    return np.concatenate((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
        d.shape[0], d.shape[1], 4, 4
    )

cpdef compute_prior_gamma_dir_np(
        blen,
        aT=1.0,
        bT=0.1,
        a=1.0,
        c=1.0,
    ):
        """Compute prior under a gamma-Dirichlet(αT , βT , α, c) prior.

        Rannala et al., 2012; Zhang et al., 2012
        Following MrBayes:
        "The prior assigns a gamma(αT , βT ) distribution for the tree length
        (sum of branch lengths), and a Dirichlet(α, c) prior for the proportion
        of branch lengths to the tree length. In the Dirichlet, the parameter for
        external branches is α and for internal branches is αc, so that the prior
        ratio between internal and external branch is c."

        Args:
            blen ([type]): [description]
            aT ([type], optional): [description]. Defaults to 1.0.
            bT ([type], optional): [description]. Defaults to 0.1.
            a ([type], optional): [description]. Defaults to 1.0.
            c ([type], optional): [description]. Defaults to 1.0.

        Returns:
            tensor: The log probability of the branch lengths under the prior.
        """
        n_branch = int(len(blen))
        n_leaf = int(n_branch / 2 + 1)

        # Dirichlet prior
        ln_prior = LogDirPrior(blen, aT, bT, a, c)

        # with prefactor
        lgamma = math.lgamma
        ln_prior = (
            ln_prior
            + (aT) * np.log(bT)
            - lgamma(aT)
            + lgamma(a * n_leaf + a * c * (n_leaf - 3))
            - n_leaf * lgamma(a)
            - (n_leaf - 3) * lgamma(a * c)
        )

        # uniform prior on topologies
        ln_prior = ln_prior - sum(np.log(np.arange(n_leaf * 2 - 5, 0, -2)))

        return ln_prior

cdef LogDirPrior(blen, aT, bT, a, c):

    n_branch = int(len(blen))
    n_leaf = int(n_branch / 2 + 1)

    treeL = np.sum(blen)

    blen[np.isclose(blen, 1.0)] = 1

    tipb = sum(np.log(blen[:n_leaf]))
    intb = sum(np.log(blen[n_leaf:]))

    lnPrior = (a - 1) * tipb + (a * c - 1) * intb
    lnPrior = (
        lnPrior
        + (aT - a * n_leaf - a * c * (n_leaf - 3)) * np.log(treeL)
        - bT * treeL
    )

    return lnPrior

cpdef compute_branch_lengths_np(
        S, peel, leaf_r, leaf_dir, int_r, int_dir, curvature=-1.0
    ):
        """Computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball

        Args:
            S (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            leaf_r ([type]): [description]
            leaf_dir ([type]): [description]
            int_r ([type]): [description]
            int_dir ([type]): [description]

        Returns:
            [type]: [description]
        """
        DTYPE = np.double
        bcount = 2 * S - 2
        blens = np.empty(bcount, dtype=np.float64)
        cdef double eps = np.finfo(np.double).eps

        for b in range(S - 1):
            directional2 = int_dir[
                peel[b][2] - S - 1,
            ]
            r2 = int_r[peel[b][2] - S - 1]

            for i in range(2):
                if peel[b][i] < S:
                    # leaf to internal
                    r1 = leaf_r[peel[b][i]]
                    directional1 = leaf_dir[peel[b][i], :]
                else:
                    # internal to internal
                    r1 = int_r[peel[b][i] - S - 1]
                    directional1 = int_dir[
                        peel[b][i] - S - 1,
                    ]

                hd = hyperbolic_distance(
                        r1, r2, directional1, directional2, curvature
                    )

                # apply the inverse transform from Matsumoto et al 2020
                hd = np.log(np.cosh(hd))

                # add a tiny amount to avoid zero-length branches
                blens[peel[b][i]] = np.clip(hd, eps, None)

        return blens