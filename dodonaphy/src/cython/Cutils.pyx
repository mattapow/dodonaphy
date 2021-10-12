import torch as torch
import numpy as np
cimport numpy as np
import scipy.spatial.distance


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

cpdef get_pdm_torch(leaf_r, leaf_dir, int_r=None, int_dir=None, curvature=-torch.ones(1)):
    """Pair-wise hyperbolic distance matrix

        Note if curvature=0, then the SQUARED Euclidean distance is computed.
    Args:
        leaf_r (tensor):
        leaf_dir (tensor):
        int_r (1D tensor):
        inr_dir (1D tensor):
        curvature (double): curvature

    Returns:
        ndarray: distance between point 1 and point 2
    """
    cdef int leaf_node_count = leaf_r.shape[0]
    cdef int int_node_count = 0
    if int_r is None:
        int_r = torch.tensor((0)).unsqueeze(dim=-1)
        int_dir = torch.tensor((0, 0)).unsqueeze(dim=-1)
    else:
        int_node_count = int_r.shape[0]
    cdef int node_count = leaf_node_count + int_node_count
    
    # return tensor
    cdef pdm = torch.zeros((node_count, node_count)).double()

    cdef int i_node
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if i < leaf_node_count and j >= leaf_node_count and int_r is not None:
                # leaf to internal
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    int_r[j_node],
                    leaf_dir[i],
                    int_dir[j_node],
                    curvature)
            elif i < leaf_node_count and j < leaf_node_count:
                # leaf to leaf
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    leaf_r[j],
                    leaf_dir[i],
                    leaf_dir[j],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance(
                    int_r[i_node],
                    int_r[j_node],
                    int_dir[i_node],
                    int_dir[j_node],
                    curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = torch.log(torch.cosh(dist_ij))

            pdm[i, j] = pdm[j, i] = dist_ij

    return pdm


cpdef get_pdm_np(leaf_r, leaf_dir, int_r=None, int_dir=None, curvature=-torch.ones(1), dtype='dict'):
    """Pair-wise hyperbolic distance matrix

        Note if curvature=0, then the SQUARED Euclidean distance is computed.
    Args:
        leaf_r (tensor):
        leaf_dir (tensor):
        int_r (1D tensor):
        inr_dir (1D tensor):
        curvature (double): curvature
        dtype (string): "dict" or "numpy"

    Returns:
        ndarray: distance between point 1 and point 2
    """
    DTYPE=np.double
    if np.isclose(curvature.detach().numpy(), np.zeros(1).astype(DTYPE)):
        # Euclidean distance
        assert dtype=='numpy', "Euclidean distances returned as numpy array. Set asNumpy to True."
        X = leaf_r[0] * leaf_dir
        pdm_linear = scipy.spatial.distance.pdist(X.detach().numpy(), metric='euclidean')
        # convert to matrix and square distances
        return scipy.spatial.distance.squareform(pdm_linear**2)

    assert dtype in ('dict', 'numpy')

    cdef np.ndarray[np.double_t, ndim=1] leaf_r_np = leaf_r.detach().numpy().astype(DTYPE)
    cdef np.ndarray[np.double_t, ndim=2] leaf_dir_np = leaf_dir.detach().numpy().astype(DTYPE)
    cdef int leaf_node_count = leaf_r.shape[0]
    cdef int int_node_count = 0
    if int_r is None:
        int_r = torch.tensor((0)).unsqueeze(dim=-1)
        int_dir = torch.tensor((0, 0)).unsqueeze(dim=-1)
    else:
        int_node_count = int_r.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] int_r_np = int_r.detach().numpy().astype(DTYPE)
    cdef np.ndarray[np.double_t, ndim=2] int_dir_np = int_dir.detach().numpy().astype(DTYPE)
    
    cdef int node_count = leaf_node_count + int_node_count
    
    # return array if pairwise distance if asNumpy
    cdef asNumpy = dtype == 'numpy'
    cdef np.ndarray[np.double_t, ndim=2] pdm_np = np.zeros((node_count*asNumpy, node_count*asNumpy))

    # return dict of lists
    if dtype == 'dict':
        pdm_dict = dict()
        for i in range(node_count):
            pdm_dict[i] = list()

    cdef double dist_ij = 0
    cdef int i_node
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if i < leaf_node_count and j >= leaf_node_count and int_r is not None:
                # leaf to internal
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance_np(
                    leaf_r_np[i],
                    int_r_np[j_node],
                    leaf_dir_np[i],
                    int_dir_np[j_node],
                    curvature)
            elif i < leaf_node_count and j < leaf_node_count:
                # leaf to leaf
                dist_ij = hyperbolic_distance_np(
                    leaf_r_np[i],
                    leaf_r_np[j],
                    leaf_dir_np[i],
                    leaf_dir_np[j],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                j_node = j - leaf_node_count
                dist_ij = hyperbolic_distance_np(
                    int_r_np[i_node],
                    int_r_np[j_node],
                    int_dir_np[i_node],
                    int_dir_np[j_node],
                    curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = np.log(np.cosh(dist_ij))

            if dtype == 'dict':
                pdm_dict[i].append(Cu_edge(dist_ij, i, j))
                pdm_dict[j].append(Cu_edge(dist_ij, j, i))
            elif dtype == 'numpy':
                pdm_np[i, j] = pdm_np[j, i] = dist_ij

    if dtype == 'dict':
        return pdm_dict
    elif dtype == 'numpy':
        return pdm_np

cpdef hyperbolic_distance_np(double r1, double r2, np.ndarray[np.double_t, ndim=1] directional1,
                            np.ndarray[np.double_t, ndim=1] directional2, double curvature):
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
    assert curvature < 0

    # Use lorentz distance for numerical stability
    cdef double eps = 0.0000000000000003
    cdef np.ndarray[np.double_t, ndim=1] z1 = poincare_to_hyper_np(dir_to_cart(r1, directional1))
    cdef np.ndarray[np.double_t, ndim=1] z2 = poincare_to_hyper_np(dir_to_cart(r2, directional2))
    cdef double inner = np.maximum(-lorentz_product_np(z1, z2), 1+eps)
    return 1. / np.sqrt(-curvature) * np.arccosh(inner)
    

cpdef hyperbolic_distance(r1, r2, directional1, directional2, curvature=-torch.ones(1)):
    """Generates hyperbolic distance between two points in poincare ball.

    Args:
        r1 (tensor): radius of point 1
        r2 (tensor): radius of point 2
        directional1 (1D tensor): directional of point 1
        directional2 (1D tensor): directional of point 2
        curvature (tensor): curvature

    Returns:
        tensor: distance between point 1 and point 2
    """
    x1 = dir_to_cart(r1, directional1)
    x2 = dir_to_cart(r2, directional2)

    if abs(curvature + 1.) > .000000001:
        return hyperbolic_distance_lorentz(x1, x2, curvature)

    invariant = 2 * torch.sum((x2-x1)**2) / (1-torch.linalg.norm(x1)**2) / (1-torch.linalg.norm(x2)**2)
    if torch.isnan(invariant):
        return hyperbolic_distance_lorentz(x1, x2, curvature)
    return torch.acosh(1 + invariant)


cpdef hyperbolic_distance_lorentz(x1, x2, curvature=-torch.ones(1)):
    """Generates hyperbolic distance between two points in poincare ball.
    Project onto hyperboloid and compute using Lorentz product.

    Returns:
        tensor: distance between point 1 and point 2
    """

    if torch.isclose(curvature, torch.zeros(1)):
        return torch.norm(x2-x1)

    z1 = poincare_to_hyper(x1).squeeze()
    z2 = poincare_to_hyper(x2).squeeze()
    eps = torch.finfo(torch.float64).eps
    inner = torch.clamp(-lorentz_product(z1, z2), min=1.+eps)
    return 1. / torch.sqrt(-curvature) * torch.acosh(inner)

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
    out[0] = (1 + b) / (1 - b + eps)
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


cpdef real2ball_LADJ(y, radius=1):
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