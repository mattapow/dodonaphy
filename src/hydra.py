import math
import warnings

from src import Cutils
import numpy as np
import torch


def hydra(D, dim=2, curvature=-1., alpha=1.1, equi_adj=0.5, **kwargs):
    """Strain minimised hyperbolic embedding
    Python Implementation of Martin Keller-Ressel's 2019 CRAN function
    hydra
    https://arxiv.org/abs/1903.08977

    Parameters
    ----------
    D : ndarray
        Pairwise distance matrix.
    dim : Int, optional
        Embedding dimension. The default is 2.
    curvature : Float, optional
        Embedding curvature. The default is -1.
    alpha : Float, optional
        Adjusts the hyperbolic curvature. Values larger than one yield a
        more distorted embedding where points are pushed to the outer
        boundary (i.e. the ideal points) of hyperblic space. The
        interaction between code{curvature} and code{alpha} is non-linear.
        The default is 1.1.
    equi_adj : Float, optional
        Equi-angular adjustment; must be a real number between zero and
        one; only used if dim is 2. Value 0 means no ajustment, 1
        adjusts embedded data points such that their angular coordinates
        in the Poincare disc are uniformly distributed. Other values
        interpolate between the two extremes. Setting the parameter to non-
        zero values can make the embedding result look more harmoniuous in
        plots. The default is 0.5.
    **kwargs :
        polar :
            Return polar coordinates in dimension 2. This flag is
            ignored in higher dimension).
        isotropic_adj :
            Perform isotropic adjustment, ignoring Eigenvalues
            (default: TRUE if dim is 2, FALSE else)
        lorentz :
            Return raw Lorentz coordinates (before projection to
            hyperbolic space) (default: FALSE)
        stress :
            Return embedding stress


    Yields
    ------
    An dictionary with:
        r : ndarray
            1D array of the radii of the embeded points
        direction : ndarray
            dim-1 array of the directions of the embedded points
        theta : ndarray
            1D array of the polar coordinate angles of the embedded points
            only if embedded into 2D Poincare disk
        stress : float
            The stress of the embedding

    """

    # sanitize/check input
    if any(np.diag(D) != 0):  # non-zero diagonal elements are set to zero
        np.fill_diagonal(D, 0)
        warnings.warn("Diagonal of input matrix D has been set to zero")

    if dim > len(D):
        raise RuntimeError("Hydra cannot embed %d points in %d-dimensions. Limit of %d." % (len(D), dim, len(D)))

    if not np.allclose(D, np.transpose(D)):
        warnings.warn(
            "Input matrix D is not symmetric.\
                Lower triangle part is used.")

    if dim == 2:
        # set default values in dimension 2
        if "isotropic_adj" not in kwargs:
            kwargs['isotropic_adj'] = True
        if "polar" in kwargs:
            kwargs['polar'] = True
    else:
        # set default values in dimension > 2
        if "isotropic_adj" in kwargs:
            kwargs['isotropic_adj'] = False
        if "polar" in kwargs:
            warnings.warn("Polar coordinates only valid in dimension two")
            kwargs['polar'] = False
        if equi_adj != 0.0:
            warnings.warn(
                "Equiangular adjustment only possible in dimension two.")

    # convert distance matrix to 'hyperbolic Gram matrix'
    A = np.cosh(np.sqrt(-curvature) * D)
    n = A.shape[0]

    # check for large/infinite values
    A_max = np.amax(A)
    if A_max > 1e8:
        warnings.warn(
            "Gram Matrix contains values > 1e8. Rerun with smaller\
            curvature parameter or rescaled distances.")
    if A_max == float("inf"):
        warnings.warn(
            "Gram matrix contains infinite values.\
            Rerun with smaller curvature parameter or rescaled distances.")

    # Compute Eigendecomposition of A
    w, v = np.linalg.eigh(A)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    # Extract leading Eigenvalue and Eigenvector
    lambda0 = w[0]
    x0 = v[:, 0]

    # Extract lower tail of spectrum)
    X = v[:, (n - dim):n]  # Last dim Eigenvectors
    spec_tail = w[(n - dim):n]  # Last dim Eigenvalues
    # A_frob = np.sqrt(np.sum(v**2)) # Frobenius norm of A

    x0 = x0 * np.sqrt(lambda0)  # scale by Eigenvalue
    if x0[0] < 0:
        x0 = -x0  # Flip sign if first element negative
    x_min = min(x0)  # find minimum

    # no isotropic adjustment: rescale Eigenvectors by Eigenvalues
    if not kwargs.get('isotropic_adj'):
        if np.array([spec_tail > 0]).any():
            warnings.warn(
                "Spectral Values have been truncated to zero. Try to use\
                lower embedding dimension")
            spec_tail[spec_tail > 0] = 0
        X = np.matmul(X, np.diag(np.sqrt(-spec_tail)))

    s = np.sqrt(np.sum(X ** 2, axis=1))
    directional = X / s[:, None]  # convert to directional coordinates

    output = {}  # Allocate output list

    # Calculate radial coordinate
    # multiplicative adjustment (scaling)
    r = np.sqrt((alpha * x0 - x_min) / (alpha * x0 + x_min))
    output['r'] = r

    # Calculate polar coordinates if dimension is 2
    if dim == 2:
        # calculate polar angle
        theta = np.arctan2(X[:, 0], -X[:, 1])

        # Equiangular adjustment
        if equi_adj > 0.0:
            angles = [(2 * x / n - 1) * math.pi for x in range(0, n)]
            theta_equi = np.array([x for _, x in sorted(
                zip(theta, angles))])  # Equi-spaced angles
            # convex combination of original and equi-spaced angles
            theta = (1 - equi_adj) * theta + equi_adj * theta_equi
            # update directional coordinate
            directional = np.array(
                [np.cos(theta), np.sin(theta)]).transpose()

            output['theta'] = theta

    output['directional'] = directional

    # Set Additional return values
    if kwargs.get('lorentz'):
        output['x0'] = x0
        output['X'] = X

    if kwargs.get('stress'):
        output['stress'] = stress(r, directional, curvature, D)

    output['curvature'] = curvature
    output['dim'] = dim
    return output


def stress(r, directional, curvature, D):
    # Calculate stress of embedding from radial/directional coordinate
    # From Nagano 2019
    n = len(r)  # number of embeded points
    stress_sq = 0.0  # allocate squared stress

    # convert from numpy to torch
    dist = torch.zeros((n, n))
    D = torch.tensor(D)

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = Cutils.hyperbolic_distance_np(
                    r[i], r[j],
                    directional[i, ], directional[j, ],
                    curvature)
                stress_sq = stress_sq + (dist[i][j] - D[i, j]) ** 2

    return np.sqrt(stress_sq)
