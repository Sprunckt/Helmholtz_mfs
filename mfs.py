import numpy as np
from scipy.special import hankel1, jv


def fundamental_solution(dist, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    return (1j/4) * hankel1(0, k*dist)


def fundamental_solution_derivative(dist, k):
    """Compute the derivative of the fundamental solution at the points given the distances,
    return a matrix of shape npoints, nsources."""
    return -(1j/4) * k * hankel1(1, k*dist)


def particular_solution(dist, theta, k, mu):
    mu = np.reshape(mu, (1, -1))
    return jv(mu, k*dist) * np.cos(mu*theta)


def particular_solution_gradient(dist, theta, k, mu, polar_offset):
    """Compute the gradient of the particular solution at the points given the distances, angles theta (both shape
    (npoints, nsrc)) and wavenumber, return a matrix of shape (npoints, nsources, 2).
    mu and polar_offset are of shape (nsources,)"""
    mu, polar_offset = np.reshape(mu, (1, -1)), np.reshape(polar_offset, (1, -1))
    dr = k*(jv(mu-1, k*dist) - jv(mu+1, k*dist)) * np.cos(mu*theta) / 2.
    dtheta = - mu*np.sin(mu*theta)*jv(mu, k*dist)

    ctheta, stheta = np.cos(theta + polar_offset), np.sin(theta + polar_offset)
    return np.stack([dr*ctheta - stheta*dtheta/dist, dr*stheta + ctheta*dtheta/dist], axis=-1)


def particular_solution_tangential_gradient(k, mu, angles, edges):
    """Compute the limit of the gradient of the particular solution at a vertex given the angles at the vertices.
    Return an array of shape (nedges, 2, 2) (res[i][j] is the gradient of the particular solution at the vertex i, j=0
    for the left edge, j=1 for the right edge)."""
    der = k*(jv(mu-1, 0.))  # in case mu=1, otherwise = 0
    res = np.zeros((len(edges), 2, 2))
    res[:, 0, :] = - np.roll(edges, 1, axis=0) * np.reshape(der*np.cos(mu*angles), [-1, 1]) / 2.
    res[:, 1, :] = edges * np.reshape(der, [-1, 1]) / 2.
    return res


def robin_operator(points, normvec, alpha, sourcesv, k, beta):
    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)

    vcontr = ((np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) / norms *  # (np, 2)*(np, npv, 2) -> (npv,)
               fundamental_solution_derivative(norms, k) + fundamental_solution(norms, k) *
               beta * 1j * k) * np.reshape(alpha, (1, -1)))

    return vcontr


class SampledBoundary:
    """Class to sample a boundary and compute the normal vector at the sampled points."""
    def __init__(self, poly, sample_param, dx=0., min_dist=0.01, sample_type="separation"):
        if sample_type == 'per_wavelength':  # sample nsrc points per wavelength where sample_param=(nsrc, k)
            src_sep = 2*np.pi/sample_param[1]/(sample_param[0] - 1)
            self.bound, self.normvec, self.bound_ids = poly.sample(src_sep, 
                                                                   min_dist=min_dist, sample_type="separation")
        elif sample_type == 'no_resample':
            self.bound, self.normvec, self.bound_ids = sample_param
        else:
            self.bound, self.normvec, self.bound_ids = poly.sample(sample_param, 
                                                                   min_dist=min_dist, sample_type=sample_type)
        self.shift(dx)

    def shift(self, dx):
        self.bound += dx*self.normvec
        return self

    def __len__(self):
        return len(self.bound)

    def copy(self):
        return SampledBoundary(None, [self.bound.copy(), self.normvec.copy(), self.bound_ids.copy()],
                               sample_type='no_resample')


