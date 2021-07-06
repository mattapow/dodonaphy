import torch as torch
import numpy as np
cimport numpy as np


cdef class Cu_edge:
    
    cdef readonly double distance
    cdef readonly int from_
    cdef readonly int to_

    def __init__(self, double distance, int node_1, int node_2):
        self.distance = distance
        self.from_ = node_1
        self.to_ = node_2

    def __lt__(self, other):
        return self.distance < other.distance

cpdef get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=1.):
    """Pair-wise hyperbolic distance matrix

    Args:
        leaf_r (tensor):
        leaf_dir (tensor):
        int_r (1D tensor):
        inr_dir (1D tensor):
        curvature (double): curvature

    Returns:
        ndarray: distance between point 1 and point 2
    """
    DTYPE=np.double
    cdef np.ndarray[np.double_t, ndim=1] leaf_r_np = leaf_r.detach().numpy().astype(DTYPE)
    cdef np.ndarray[np.double_t, ndim=2] leaf_dir_np = leaf_dir.detach().numpy().astype(DTYPE)
    cdef np.ndarray[np.double_t, ndim=1] int_r_np = int_r.detach().numpy().astype(DTYPE)
    cdef np.ndarray[np.double_t, ndim=2] int_dir_np = int_dir.detach().numpy().astype(DTYPE)
    cdef int leaf_node_count = leaf_r.shape[0]
    cdef int node_count = leaf_r.shape[0] + int_r.shape[0]
    edge_list = dict()
    for i in range(node_count):
        edge_list[i] = list()

    cdef double dist_ij = 0
    cdef int i_node
    for i in range(node_count):
        for j in range(max(i + 1, leaf_node_count), node_count):

            if (i < leaf_node_count):
                # leaf to internal
                dist_ij = hyperbolic_distance_np(
                    leaf_r_np[i],
                    int_r_np[j - leaf_node_count],
                    leaf_dir_np[i],
                    int_dir_np[j - leaf_node_count],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                dist_ij = hyperbolic_distance_np(
                    int_r_np[i_node],
                    int_r_np[j - leaf_node_count],
                    int_dir_np[i_node],
                    int_dir_np[j - leaf_node_count],
                    curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = np.log(np.cosh(dist_ij))

            edge_list[i].append(Cu_edge(dist_ij, i, j))
            edge_list[j].append(Cu_edge(dist_ij, j, i))

    return edge_list

cpdef hyperbolic_distance_np(double r1, double r2, np.ndarray[np.double_t, ndim=1] directional1,
                            np.ndarray[np.double_t, ndim=1] directional2,double curvature):
    """Generates hyperbolic distance between two points in poincoire ball

    Args:
        r1 (ndarray): radius of point 1
        r2 (ndarray): radius of point 2
        directional1 (1D ndarray): directional of point 1
        directional2 (1D ndarray): directional of point 2
        curvature (ndarray): curvature

    Returns:
        ndarray: distance between point 1 and point 2
    """
    # if torch.allclose(r1, r2) and torch.allclose(directional1, directional2):
    #     return torch.zeros(1)

    # Use lorentz distance for numerical stability
    cdef double eps = 0.0000000000000003
    cdef np.ndarray[np.double_t, ndim=1] z1 = poincare_to_hyper_np(dir_to_cart(r1, directional1))
    cdef np.ndarray[np.double_t, ndim=1] z2 = poincare_to_hyper_np(dir_to_cart(r2, directional2))
    cdef double inner = np.maximum(-lorentz_product_np(z1, z2), 1+eps)
    return 1. / np.sqrt(curvature) * np.arccosh(inner)
    

cpdef hyperbolic_distance(r1, r2, directional1, directional2, curvature):
    """Generates hyperbolic distance between two points in poincoire ball

    Args:
        r1 (tensor): radius of point 1
        r2 (tensor): radius of point 2
        directional1 (1D tensor): directional of point 1
        directional2 (1D tensor): directional of point 2
        curvature (tensor): curvature

    Returns:
        tensor: distance between point 1 and point 2
    """
    # if torch.allclose(r1, r2) and torch.allclose(directional1, directional2):
    #     return torch.zeros(1)

    # Use lorentz distance for numerical stability
    cdef double eps = 0.0000000000000003
    z1 = poincare_to_hyper(dir_to_cart(r1, directional1))
    z2 = poincare_to_hyper(dir_to_cart(r2, directional2))
    inner = torch.clamp(-lorentz_product(z1, z2), min=1+eps)
    return 1. / torch.sqrt(curvature) * torch.arccosh(inner)


cdef lorentz_product_np(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : ndarray
        1D array of a point on the hyperboloid.
    y : ndarray optional
        1D array of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    return -x[0] * y[0] + np.dot(x[1:], y[1:])

cpdef lorentz_product(x, y=None):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : tensor
        1D array of a point on the hyperboloid.
    y : tensor optional
        1D array of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    if y is None:
        y = x
    if type(x).__module__ == 'torch':
        return -x[0] * y[0] + torch.dot(x[1:], y[1:])
    elif type(x).__module__ == 'numpy':
        return -x[0] * y[0] + np.dot(x[1:], y[1:])
    raise TypeError('x must be numpy or torch')
    
cdef poincare_to_hyper_np(np.ndarray[np.double_t, ndim=1] location):
    """
    Take point in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: ndarray
        location of point in poincare ball

    Returns
    -------

    """
    DTYPE = np.double
    cdef double eps = 0.0000000000000003
    cdef int dim = location.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(dim + 1)
    cdef np.ndarray[np.double_t, ndim=1] a = np.power(location[:], 2)
    cdef double b = a.sum(axis=0)
    out[0] = (1 + b) / (1 - b)
    out[1:] = 2 * location[:] / (1 - b + eps)
    return out

cpdef poincare_to_hyper(location):
    """
    Take points in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: tensor
        n_points x dim location of points in poincare ball

    Returns
    -------

    """
    cdef double eps = 0.0000000000000003
    cdef int dim
    if location.ndim == 1:
        dim = location.shape[0]
        out = torch.zeros(dim + 1, dtype=torch.double)
        a = location[:].pow(2).sum(dim=0)
        out[0] = (1 + a) / (1 - a)
        out[1:] = 2 * location[:] / (1 - a + eps)
    elif location.ndim == 2:
        dim = location.shape[1]
        a = location.pow(2).sum(dim=-1)
        out0 = torch.div((1 + a), (1 - a))
        out1 = 2 * location / (1 - a.unsqueeze(dim=1) + eps)
        out = torch.cat((out0.unsqueeze(dim=1), out1), dim=1)
    return out

cdef dir_to_cart(r, directional):
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
