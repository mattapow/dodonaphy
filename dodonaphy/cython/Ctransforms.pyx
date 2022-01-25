import numpy as np
cimport numpy as np
import torch

cdef np.double_t eps = np.finfo(np.double).eps

cpdef ball2real(np.ndarray[np.double_t, ndim=2] loc_ball, np.double_t radius=1.0):
    """A map from the unit ball B^n to real R^n.
    Inverse of real2ball.

    Args:
        loc_ball (tensor): [description]
        radius (tensor): [description]

    Returns:
        tensor: [description]
    """
    if loc_ball.ndim == 1:
        loc_ball = np.expand_dims(loc_ball, -1)
    cdef np.int_t dim = loc_ball.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] norm_loc_ball = np.tile(np.linalg.norm(loc_ball, axis=-1, keepdims=True), (1, dim))
    cdef np.ndarray[np.double_t, ndim=2] loc_real = loc_ball / (radius - norm_loc_ball)
    return loc_real


cpdef real2ball(np.ndarray[np.double_t, ndim=2] loc_real, np.double_t radius=1.0):
    """A map from the reals R^n to unit ball B^n.
    Inverse of ball2real.

    Args:
        loc_real (tensor): Point in R^n with size [1, dim]
        radius (float): Radius of ball

    Returns:
        tensor: [description]
    """
    if loc_real.ndim == 1:
        loc_real = np.expand_dims(loc_real, -1)
    cdef np.int_t dim = loc_real.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] norm_loc_real = np.tile(np.linalg.norm(loc_real, axis=-1, keepdims=True), (1, dim))
    cdef np.ndarray[np.double_t, ndim=2] loc_ball = np.divide(radius * loc_real, (1 + norm_loc_real))
    return loc_ball


cpdef real2ball_LADJ(np.ndarray[np.double_t, ndim=2] y, np.double_t radius=1.0):
    """Copmute log of absolute value of determinate of jacobian of real2ball on point y

    Args:
        y (tensor): Points in R^n n_points x n_dimensions

    Returns:
        scalar tensor: log absolute determinate of Jacobian
    """
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    n, D = np.shape(y)
    cdef np.double_t log_abs_det_J = 0.0
    cdef np.ndarray[np.double_t, ndim=2] norm = np.linalg.norm(y, axis=-1, keepdims=True)
    cdef np.ndarray[np.double_t, ndim=2] J = np.zeros((D, D))
    for k in range(n):
        J = (np.eye(D, D) - np.outer(y[k], y[k]) / (norm[k] * (norm[k] + 1))) / (1+norm[k])
        sign, log_det = np.linalg.slogdet(radius * J)
        log_abs_det_J = log_abs_det_J + sign*log_det
    return log_abs_det_J

cpdef ball2real_torch(loc_ball, radius=1):
    """A map from the unit ball B^n to real R^n.
    Inverse of real2ball.

    Args:
        loc_ball (tensor): [description]
        radius (tensor): [description]

    Returns:
        tensor: [description]
    """
    if loc_ball.ndim == 1:
        loc_ball = loc_ball.unsqueeze(dim=-1)
    dim = loc_ball.shape[1]
    norm_loc_ball = torch.norm(loc_ball, dim=-1, keepdim=True).repeat(1, dim)
    loc_real = loc_ball / (radius - norm_loc_ball)
    return loc_real


cpdef real2ball_torch(loc_real, radius=1):
    """A map from the reals R^n to unit ball B^n.
    Inverse of ball2real.

    Args:
        loc_real (tensor): Point in R^n with size [1, dim]
        radius (float): Radius of ball

    Returns:
        tensor: [description]
    """
    if loc_real.ndim == 1:
        loc_real = loc_real.unsqueeze(dim=-1)
    dim = loc_real.shape[1]

    norm_loc_real = torch.norm(loc_real, dim=-1, keepdim=True).repeat(1, dim)
    loc_ball = torch.div(radius * loc_real, (1 + norm_loc_real))
    return loc_ball



cpdef real2ball_LADJ_torch(y, radius=1.0):
    """Copmute log of absolute value of determinate of jacobian of real2ball on point y
    Args:
        y (tensor): Points in R^n n_points x n_dimensions
    Returns:
        scalar tensor: log absolute determinate of Jacobian
    """
    if y.ndim == 1:
        y = y.unsqueeze(dim=-1)
    n, D = y.shape
    log_abs_det_J = torch.zeros(1)
    norm = torch.norm(y, dim=-1, keepdim=True)
    for k in range(n):
        J = (torch.eye(D, D) - torch.outer(y[k], y[k]) / (norm[k] * (norm[k] + 1))) / (1+norm[k])
        log_abs_det_J = log_abs_det_J + torch.logdet(radius * J)
    return log_abs_det_J

