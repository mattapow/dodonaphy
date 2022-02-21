import math
import numpy as np
cimport numpy as np
from . import Chyp_np
from collections import Counter

cdef double eps = np.finfo(np.double).eps

def compute_LL_np(partials,
    np.ndarray[np.int_t, ndim=1] weights,
    np.ndarray[np.int_t, ndim=2] peel,
    np.ndarray[np.double_t, ndim=1] blen):
    """
    Compute tree likeilhood.
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
    cdef np.ndarray[np.double_t, ndim=2] d = np.expand_dims(branch_lengths, axis=1)
    cdef np.ndarray[np.double_t, ndim=2] a = 0.25 + 3.0 / 4.0 * np.exp(-4.0 / 3.0 * d)
    cdef np.ndarray[np.double_t, ndim=2] b = 0.25 - 0.25 * np.exp(-4.0 / 3.0 * d)
    return np.concatenate((a, b, b, b, b, a, b, b, b, b, a, b, b, b, b, a), -1).reshape(
        d.shape[0], d.shape[1], 4, 4
    )

cpdef compute_prior_gamma_dir_np(
        np.ndarray[np.double_t, ndim=1] blen,
        np.double_t aT=1.0,
        np.double_t bT=0.1,
        np.double_t a=1.0,
        np.double_t c=1.0,
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
        cdef int n_branch = int(len(blen))
        cdef int n_leaf = int(n_branch / 2 + 1)

        # Dirichlet prior
        cdef np.double_t ln_prior = LogDirPrior(blen, aT, bT, a, c)

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
        ln_prior = ln_prior - np.sum(np.log(np.arange(n_leaf * 2 - 5, 0, -2)))

        return ln_prior

cdef LogDirPrior(np.ndarray[np.double_t, ndim=1] blen, np.double_t aT, np.double_t bT, np.double_t a, np.double_t c):
    cdef int n_branch = int(len(blen))
    cdef int n_leaf = int(n_branch / 2 + 1)
    cdef np.double_t treeL = np.maximum(np.sum(blen), eps)
    cdef np.double_t tipb = np.sum(np.log(np.maximum(blen[:n_leaf], eps)))
    cdef np.double_t intb = np.sum(np.log(np.maximum(blen[n_leaf:], eps)))
    lnPrior = (a - 1) * tipb + (a * c - 1) * intb
    lnPrior = (
        lnPrior
        + (aT - a * n_leaf - a * c * (n_leaf - 3)) * np.log(treeL)
        - bT * treeL
    )
    return lnPrior

cpdef compute_branch_lengths_np(
        np.int_t S,
        np.ndarray[np.int_t, ndim=2] peel,
        np.ndarray[np.double_t, ndim=2] leaf_x,
        int_x,
        np.double_t curvature=-1.0,
        bint matsumoto=False
    ):
        """Computes the hyperboloid distance points in peel.

        Args:
            S (integer): [description]
            D ([type]): [description]
            peel ([type]): [description]
            leaf_x ([type]): [description]
            int_x ([type]): [description]

        Returns:
            [type]: [description]
        """
        cdef np.int_t bcount = 2 * S - 2
        cdef np.ndarray[np.double_t, ndim=1] blens = np.zeros(bcount, dtype=np.double)

        for b in range(S - 1):
            x2 = int_x[peel[b][2] - S - 1,]
            x2_sheet = Chyp_np.project_up(x2)

            for i in range(2):
                if peel[b][i] < S:
                    x1 = leaf_x[peel[b][i], :]  # leaf to internal
                else:
                    x1 = int_x[peel[b][i] - S - 1, :]  # internal to internal
                x1_sheet = Chyp_np.project_up(x1)
                hd = Chyp_np.hyperbolic_distance(x1_sheet, x2_sheet, curvature)
                if matsumoto:
                    hd = np.log(np.cosh(hd))

                # avoid zero-length branches
                blens[peel[b][i]] = np.maximum(hd, eps)

        return blens