import numpy as np
cimport numpy as np
from . import Cutils, Cphylo, Chyperboloid_np
from . import utils

cpdef sample_loc_np(
    int taxa,
    int dim,
    np.ndarray[np.double_t, ndim=2] loc,
    np.ndarray[np.double_t, ndim=2] cov,
    embedder,
    bint is_internal,
    normalise_leaf=False):
    """Given locations in poincare ball, transform them to Euclidean
    space, sample from a Normal and transform sample back."""
    cdef int n_locs
    if is_internal:
        n_locs = taxa - 2
    else:
        n_locs = taxa
    cdef int n_vars = n_locs * dim
    cdef np.ndarray[np.double_t, ndim=2] loc_t0
    cdef np.ndarray[np.double_t, ndim=2] loc_prop

    rng = np.random.default_rng()
    if embedder == "simple":
        # transform internals to R^n
        loc_t0 = Chyperboloid_np.ball2real(loc)
        log_abs_det_jacobian = -Chyperboloid_np.real2ball_LADJ(loc_t0)

        # propose new int nodes from normal in R^n
        sample = rng.multivariate_normal(loc_t0.flatten(), cov)
        loc_t0 = sample.reshape((n_locs, dim))

        # convert ints to poincare ball
        loc_prop = Chyperboloid_np.real2ball(loc_t0)

    elif embedder == "wrap":
        # transform ints to R^n
        loc_t0, jacobian = Chyperboloid_np.p2t0(loc, get_jacobian=True)
        log_abs_det_jacobian = -jacobian

        # propose new int nodes from normal in R^n
        sample = rng.multivariate_normal(
            np.zeros(n_vars, dtype=np.double), cov
        )
        loc_prop = Chyperboloid_np.t02p(
            sample.reshape(n_locs, dim),
            loc_t0.reshape(n_locs, dim),
        )

    if normalise_leaf:
        # TODO: do we need normalise jacobian? The positions are inside the integral... so yes
        r_prop = np.tile(np.linalg.norm(loc_prop[0, :], keepdims=True), (taxa))
        loc_prop = Cutils.normalise_np(loc_prop) * np.tile(r_prop, (dim, 1)).T
    else:
        r_prop = np.linalg.norm(loc_prop, axis=-1)
    cdef np.ndarray[np.double_t, ndim=2] dir_prop = loc_prop / np.linalg.norm(loc_prop, axis=1, keepdims=True)
    return r_prop, dir_prop, log_abs_det_jacobian