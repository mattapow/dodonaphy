import torch
import numpy as np
cimport numpy as np


cpdef dir_to_cart(r, directional):
        """convert radius/ directionals to cartesian coordinates [x,y,z,...]

        Parameters
        ----------
        r (1D ndarray): radius of each n_points
        directional (2D ndarray): n_points x dim directional of each point

        Returns
        -------
        (2D ndarray) Cartesian coordinates of each point n_points x dim

        """
        npScalar = type(directional).__module__ == np.__name__ and np.isscalar(r)
        torchScalar = type(directional).__module__ == torch.__name__ and (r.shape == torch.Size([]))
        if npScalar or torchScalar:
            return directional * r
        return directional * r[:, None]


def cart_to_dir_np(X):
    """convert positions in X in  R^dim to radius/ unit directional

    Parameters
    ----------
    X (ndarray): Points in R^dim

    Returns
    -------
    r (ndarray): radius
    directional (ndarray): unit vectors

    """

    if X.ndim == 1:
        X = np.expand_dims(X, 0)
    r = np.power(np.power(X, 2).sum(axis=1), 0.5)
    directional = X / r[:, None]

    for i in np.where(np.isclose(r, np.zeros_like(r))):
        directional[i, 0] = 1
        directional[i, 1:] = 0

    return r, directional

cpdef normalise_np(y):
    """Normalise vectors to unit sphere

    Args:
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    if y.size == 1:
        y = np.expand_dims(y, axis=1)
    dim = y.shape[1]

    norm = np.tile(np.linalg.norm(y, axis=-1, keepdims=True), (1, dim))
    return y / norm

def angle_to_directional_np(theta):
    """
    Convert polar angles to unit vectors

    Parameters
    ----------
    theta : ndarray
        Angle of points.

    Returns
    -------
    directional : ndarray
        Unit vectors of points.

    """
    dim = 2
    n_points = len(theta)
    directional = np.zeros((n_points, dim))
    directional[:, 0] = np.cos(theta)
    directional[:, 1] = np.sin(theta)
    return directional

cpdef dir_to_cart_np(r, directional):
    """convert radius/ directionals to cartesian coordinates [x,y,z,...]

    Parameters
    ----------
    r (1D ndarray): radius of each n_points
    directional (2D ndarray): n_points x dim directional of each point

    Returns
    -------
    (2D ndarray) Cartesian coordinates of each point n_points x dim

    """
    if np.isscalar(r):
        return directional * r
    return directional * r[:, None]
