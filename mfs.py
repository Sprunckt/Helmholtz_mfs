import numpy as np
from scipy.special import hankel1
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import gmres
from utils import Polygon
from scipy.optimize import minimize
from scipy.integrate import simpson
from scipy.sparse.linalg import cg
import time


def fundamental_solution(points, sources, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    return (1j/4) * hankel1(0, k*np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=-1))


def fundamental_solution_derivative(dist, k):
    """Compute the derivative of the fundamental solution at the points given the distances,
    return a matrix of shape npoints, nsources."""
    return -(1j/4) * k * hankel1(1, k*dist)


def compute_sol(alpha, points, sourcesv, k, source=None):
    """Compute the solution at the points given the virtual sources and the associated coefficients."""

    if source is None:
        return np.sum(np.reshape(alpha, -1)[np.newaxis, :] * fundamental_solution(points, sourcesv, k), axis=1)
    else:
        return (np.sum(np.reshape(alpha, -1)[np.newaxis, :] *
                       fundamental_solution(points, sourcesv, k), axis=1) +
                np.reshape(fundamental_solution(points, source, k), -1))


def compute_exact_sol(points, image_src, k, weights=None):
    """Compute the exact solution at the points given the source position."""
    if weights is None:
        weights = np.ones(len(image_src))
    return np.sum(weights[np.newaxis, :] * fundamental_solution(points, image_src, k), axis=1)


def solve(beta, normvec, bound, boundv, source, k, source_type='dirac', cond='dirichlet', method='direct',
          return_mat=False):
    """Solve the Helmholtz equation with the given boundary conditions and return the coefficients of the solution in
    the MFS.
    args: - beta: impedance at the boundary
            - normvec: normal vector to the boundary
            - bound: boundary points
            - boundv: virtual boundary points for the MFS
            - source: position of a point source (dirac source term) if source_type=='dirac', otherwise a
            renormalization point (ie. a point where the solution value is set to 1) to compute eigenmodes. In the case
            of a dirac source, the source term in the equation is -delta(x-x0), where x0 is the source position.
            - source_type: 'dirac' or 'renormalization'. If set to 'renormalization', the solution is set to 1 at the
            renormalization point, and the last boundary point of the vector is ignored to match the number of points
            and virtual sources.
            function if it is a callable (accepts a (n,2) shaped array of points an a float k and returns a (n,) shaped
            array of values)
            - k: wavenumber
            - cond: boundary condition, 'dirichlet', 'neumann', 'robin'
    """

    # assemble the system matrix and rhs
    npoints, nsrc = len(bound), len(boundv)
    A, b = np.zeros((npoints, nsrc), dtype=complex), np.zeros((npoints, 1), dtype=complex)

    if source_type == 'dirac':
        indstop = npoints
    elif source_type == 'renormalization':
        indstop = -1
    else:
        raise ValueError('source_type must be "dirac" or "renormalization"')

    if cond == "neumann":
        diffs = bound[:indstop, np.newaxis, :] - boundv[np.newaxis, :, :]
        norms = np.linalg.norm(diffs, axis=-1)
        A[:indstop, :] = (np.sum(normvec[:indstop, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, np, 2) -> (np, np)
                          norms * fundamental_solution_derivative(norms, k))

        if source_type == 'dirac':  # inhomogenous equation with dirac source
            diffs = bound[:, np.newaxis, :] - source[np.newaxis, :, :]  # rhs computation
            norms = np.linalg.norm(diffs, axis=-1)
            b[:] = - (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) *
                      fundamental_solution_derivative(norms, k) / norms)

    elif cond == "dirichlet":
        A[:indstop, :] = fundamental_solution(bound[:indstop], boundv, k)
        if source_type == 'dirac':
            b[:] = - fundamental_solution(bound, source, k)

    elif cond == "robin":
        diffs = bound[:indstop, np.newaxis, :] - boundv[np.newaxis, :, :]
        norms = np.linalg.norm(diffs, axis=-1)
        A[:indstop, :] = (np.sum(normvec[:indstop, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, np, 2)-> (np, np)
                          norms * fundamental_solution_derivative(norms, k) +
                          fundamental_solution(bound[:indstop], boundv, k) * beta[:indstop, np.newaxis] * 1j * k)

        if source_type == 'dirac':
            diffs = bound[:, np.newaxis, :] - source[np.newaxis, :, :]
            norms = np.linalg.norm(diffs, axis=-1)
            b[:] = -(np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) * fundamental_solution_derivative(norms, k)
                     / norms +
                     fundamental_solution(bound, source, k) * beta[:, np.newaxis] * 1j * k)

    if source_type == 'renormalization':  # homogeneous equation with anchor point
        A[-1, :] = np.ones(npoints)
        b[-1] = 1.

    if method == "direct":
        res = np.linalg.solve(A, b)
    elif method == 'lstsq':
        res = np.linalg.lstsq(A, b, rcond=-1)[0]
    elif method == "tsvd":
        res = np.linalg.pinv(A) @ b
    elif method == "gmres":
        res = gmres(A, b, tol=1e-8)[0]
    elif method == "ridge":
        lasso_fitter = Ridge(alpha=1e-3, fit_intercept=False, tol=1e-8)
        Ac1 = np.concatenate([np.real(A), -np.imag(A)], axis=1)
        Ac2 = np.concatenate([np.imag(A), np.real(A)], axis=1)
        A = np.concatenate([Ac1, Ac2], axis=0)
        b = np.concatenate([np.real(b), np.imag(b)], axis=0)
        scale = np.sqrt(len(A))  # rescaling factor for sklearn convention
        lasso_fitter.fit(scale * A, scale * b)
        res = lasso_fitter.coef_.flatten()
        res = res[:nsrc] + 1j * res[nsrc:]
    else:
        raise ValueError("method must be 'direct', 'tsvd', 'lstsq', 'ridge' or 'gmres'")

    if return_mat:
        return res, A
    else:
        return res


def neumann_operator(points, normvec, alpha, sourcesv, k):
    """ Computes the normal derivative of the solution given by the virtual sources sourcesv with coefficients alpha at
    the points points with normal vector normvec. Return the contribution from the virtual sources.
    """

    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    vcontr = np.sum(np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, npv, 2) -> (npv,)
                    norms * fundamental_solution_derivative(norms, k)*alpha.reshape(1, -1), axis=1)
    return vcontr


def robin_operator(points, normvec, alpha, sourcesv, k, beta):
    if len(beta) == 0:
        beta = np.ones(len(points), dtype=np.complex128)*beta
    else:
        beta = np.ones(len(points), dtype=np.complex128)*beta[0]

    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    vcontr = np.sum((np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) / norms * # (np, 2)*(np, npv, 2) -> (npv,)
                     fundamental_solution_derivative(norms, k) + fundamental_solution(points, sourcesv, k) *
                     beta.reshape((-1, 1)) * 1j * k) * alpha.reshape(1, -1), axis=1)
    return vcontr


class SlidingMFS:
    def __init__(self, nbound: int, nsrc: int, nobs: int, pos_src: np.ndarray, poly: Polygon,
                 k: float, dx: float, cond='neumann', lam=0., sample_kwargs=None, beta=None):
        self.dx = dx
        self.pos_src = pos_src
        self.poly = poly
        self.k = k
        self.cond = cond
        self.nbound, self.nsrc, self.nobs = nbound, nsrc, nobs
        self.lam = lam
        if sample_kwargs is None:
            sample_kwargs = {"distribution": 'uniform', "min_dist": 0.}
        self.bound, self.normvec, self.bound_ids, self.params = poly.sample(nbound, **sample_kwargs,
                                                                            return_ids=True, return_nodes=True)
        self.bound_obs, self.normvec_obs = poly.sample(nobs, distribution='cheby')
        boundv, self.normvecv, self.boundv_ids, self.paramsv = poly.sample(nsrc, **sample_kwargs,
                                                                           return_ids=True, return_nodes=True)
        self.boundv = boundv + dx*self.normvecv

        self.boundary_operator = self.neumann_operator  # only neumann boundary conditions implemented so far
        self.y = self.boundary_operator(-np.ones(1), self.pos_src)

        if beta is None:
            self.beta = np.zeros(self.nbound)
        else:
            self.beta = beta*np.ones(self.nbound)

        self.ampl = np.zeros(self.nsrc, dtype=np.complex128)

    def gram_fit(self, verbose=True):
        """Compute scalar products between fundamental solutions, use the boundary points as quadrature points"""
        A = np.zeros((self.nsrc, self.nsrc), dtype=np.complex128)
        t1 = time.time()
        # compute the value of the normal derivative of the fundamental solution at the observation points
        if self.cond == 'neumann':
            diffs = self.bound[:, np.newaxis, :] - self.boundv[np.newaxis, :, :]  # (npo, npv, 2)
            norms = np.linalg.norm(diffs, axis=-1)  # (npo, npv)
            vcontr = (np.sum(self.normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (npo, npv)
                      norms * fundamental_solution_derivative(norms, self.k))

            # compute contribution of the source
            diffs = self.bound[:, np.newaxis, :] - self.pos_src[np.newaxis, :, :]  # (npo, 2)
            norms = np.linalg.norm(diffs, axis=-1)  # (npo, nsrc)
            scontr = (np.sum(self.normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (npo,)
                      norms * fundamental_solution_derivative(norms, self.k)).squeeze()
        elif self.cond == 'dirichlet':
            vcontr = fundamental_solution(self.bound, self.boundv, self.k)
            scontr = fundamental_solution(self.bound, self.pos_src, self.k)

        b = np.zeros((self.nsrc, 1), dtype=np.complex128)
        for i in range(self.nsrc):
            for j in range(i, self.nsrc):
                # compute L2 scalar product between green functions
                prod = vcontr[:, i].conj() * vcontr[:, j]  # (npo,)
                # apply quadrature rule (trapezoidal rule) over each edge
                for k in range(self.poly.nedges):
                    ind_edge = self.bound_ids == k
                    A[i, j] += simpson(prod[ind_edge], self.params[ind_edge])*self.poly.lengths[k]
                A[j, i] = A[i, j].conj()
            for k in range(self.poly.nedges):
                ind_edge = self.bound_ids == k
                b[i] += simpson(vcontr[ind_edge, i]*scontr[ind_edge].conj(), self.params[ind_edge])*self.poly.lengths[k]

        real_matrix = np.concatenate([np.concatenate([np.real(A), -np.imag(A)], axis=1),
                                      np.concatenate([np.imag(A), np.real(A)], axis=1)], axis=0)
        real_rhs = np.concatenate([-np.real(b), np.imag(b)])
        t2 = time.time()
        if verbose:
            print("Gram matrix computation took {:.3f} s".format(t2 - t1))
        self.ampl = np.linalg.lstsq(real_matrix, real_rhs, rcond=-1)[0]
        self.ampl = self.ampl[:self.nsrc] + 1j*self.ampl[self.nsrc:]
        t3 = time.time()
        if verbose:
            print("Solving took {:.3f} s".format(t3 - t2))
        return self.ampl, np.linalg.cond(real_matrix)

    def mfs_fit(self, solver='lstsq'):
        self.ampl, mat = solve(self.beta, self.normvec, self.bound, self.boundv, self.pos_src, self.k,
                               cond=self.cond, method=solver, return_mat=True).squeeze()

        return self.ampl, np.linalg.cond(mat)

    def neumann_operator(self, a, x):
        """Observation operator mapping virtual sources x (shape (nsrc, 2)) and amplitudes a (shape nsrc) to the
        resulting boundary values (shape nbound)"""
        return neumann_operator(self.bound_obs, self.normvec_obs, a, x, self.k).squeeze()

    def cost_function(self, var):
        a = var[:2*self.nsrc]  # amplitudes are complex valued
        a = a[:self.nsrc] + 1j*a[self.nsrc:]
        params = var[2*self.nsrc:]
        x = self.poly.parametrization_to_coord(self.boundv_ids, params) + self.dx*self.normvecv

        return 0.5*np.linalg.norm(self.boundary_operator(a, x) - self.y)**2 + self.lam*np.sum(np.abs(a))

    def src_slide_step(self, verbose=False):
        """apply lbfgs algorithm to minimize the cost function by sliding virtual sources along the virtual boundary
            and updating the coefficients of the virtual sources"""

        bounds = [(-np.inf, np.inf)]*2*self.nsrc + [(0, 1)]*self.nsrc

        ini = np.concatenate([np.real(self.ampl), np.imag(self.ampl), self.paramsv])
        print("optimizing...")
        opti_res = minimize(self.cost_function, ini, jac=None, bounds=bounds, options={'gtol':1})
        print("done")
        # update the virtual sources
        self.paramsv = opti_res.x[2*self.nsrc:]
        self.ampl = opti_res.x[:self.nsrc] + 1j*opti_res.x[self.nsrc:2*self.nsrc]
        self.boundv = self.poly.parametrization_to_coord(self.boundv_ids, self.paramsv) + self.dx*self.normvecv

        if verbose:
            val_ini = self.cost_function(ini)
            print("Optimization result: {} {}".format(opti_res.success, opti_res.message))
            print("Initial/final values: {} {}".format(val_ini, opti_res.fun))
