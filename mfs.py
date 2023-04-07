import numpy as np
from scipy.special import hankel1
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from scipy.sparse.linalg import gmres


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
            renormalization point (ie. a point where the solution value is set to 1)
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


def neumann_operator(points, normvec, alpha, sourcesv, source, k):
    """ Computes the normal derivative of the solution given by the virtual sources sourcesv with coefficients alpha at
    the points points with normal vector normvec. Return the contribution from the virtual sources and the contribution
    from the source.
    """

    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    vcontr = np.sum(np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, npv, 2) -> (npv,)
                    norms * fundamental_solution_derivative(norms, k)*alpha.reshape(1, -1), axis=1)
    diffs = points[:, np.newaxis, :] - source[np.newaxis, :, :]
    norms = np.linalg.norm(diffs, axis=-1)

    src_contr = - (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) *
                   fundamental_solution_derivative(norms, k) / norms)

    return vcontr, src_contr
