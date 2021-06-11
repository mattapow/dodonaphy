import torch

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

cpdef get_pdm(leaf_r, leaf_dir, int_r, int_dir, curvature=torch.ones(1)):
    leaf_node_count = leaf_r.shape[0]
    node_count = leaf_r.shape[0] + int_r.shape[0]
    edge_list = dict()
    for i in range(node_count):
        edge_list[i] = list()

    for i in range(node_count):
        for j in range(max(i + 1, leaf_node_count), node_count):
            dist_ij = 0

            if (i < leaf_node_count):
                # leaf to internal
                dist_ij = hyperbolic_distance(
                    leaf_r[i],
                    int_r[j - leaf_node_count],
                    leaf_dir[i],
                    int_dir[j - leaf_node_count],
                    curvature)
            else:
                # internal to internal
                i_node = i - leaf_node_count
                dist_ij = hyperbolic_distance(
                    int_r[i_node],
                    int_r[j - leaf_node_count],
                    int_dir[i_node],
                    int_dir[j - leaf_node_count],
                    curvature)

            # apply the inverse transform from Matsumoto et al 2020
            dist_ij = torch.log(torch.cosh(dist_ij))

            edge_list[i].append(Cu_edge(dist_ij, i, j))
            edge_list[j].append(Cu_edge(dist_ij, j, i))

    return edge_list

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
    z1 = poincare_to_hyper(dir_to_cart(r1, directional1)).squeeze()
    z2 = poincare_to_hyper(dir_to_cart(r2, directional2)).squeeze()
    eps = torch.finfo(torch.float64).eps
    inner = torch.clamp(-lorentz_product(z1, z2), min=1+eps)
    return 1. / torch.sqrt(curvature) * torch.acosh(inner)


cdef lorentz_product(x, y=None):
    """
    The lorentzian product of x and y

    This can serve as a metric, i.e. distance function between points.
    lorentz_product(x)=-1 iff x is on the hyperboloid

    Parameters
    ----------
    x : Tensor
        1D tensor of a point on the hyperboloid.
    y : Tensor optional
        1D tensor of a point on the hyperboloid. The default is None.

    Returns
    -------
        Lorentzian product of x and y.

    """
    if y is None:
        y = x
    return -x[0] * y[0] + torch.dot(x[1:], y[1:])

cdef poincare_to_hyper(location):
    """
    Take points in Poincare ball to hyperbolic sheet

    Parameters
    ----------
    location: Tensor
        n_points x dim location of points in poincare ball

    Returns
    -------

    """
    eps = torch.finfo(torch.double).eps
    if location.ndim == 1:
        dim = len(location)
        out = torch.zeros(dim + 1).double()
        a = location[:].pow(2).sum(0)
        out[0] = (1 + a) / (1 - a)
        out[1:] = 2 * location[:] / (1 - a + eps)

    elif location.ndim == 2:
        dim = location.shape[1]
        a = location.pow(2).sum(-1)
        out0 = torch.div((1 + a), (1 - a))
        out1 = 2 * location / (1 - a.unsqueeze(dim=1) + eps)
        out = torch.cat((out0.unsqueeze(dim=1), out1), dim=1)
    return out

cdef dir_to_cart(r, directional):
        """convert radius/ directionals to cartesian coordinates [x,y,z,...]

        Parameters
        ----------
        r (1D tensor): radius of each n_points
        directional (2D tensor): n_points x dim directional of each point

        Returns
        -------
        (2D tensor) Cartesian coordinates of each point n_points x dim

        """

        if r.shape == torch.Size([]):
            return directional * r
        return directional * r[:, None]