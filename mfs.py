import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from utils import out_polygon
from shapely.geometry import Polygon
from sklearn.linear_model import Lasso


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


def solve(beta, normvec, bound, boundv, source, k, source_type='dirac', cond='dirichlet'):
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
    npoints = len(bound)
    A, b = np.zeros((npoints, npoints), dtype=complex), np.zeros((npoints, 1), dtype=complex)

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

    return np.linalg.solve(A, b)


def neumann_operator(bound, normvec, boundv, k):
    """Compute the neumann operator at the boundary points given the virtual boundary points."""
    diffs = bound[:, np.newaxis, :] - boundv[np.newaxis, :, :]
    norms = np.linalg.norm(diffs, axis=-1)
    return (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, np, 2) -> (np, np)
                      norms * fundamental_solution_derivative(norms, k))


def fit_weights(y, target, boundary, normvec, b_operator, k, v_src, lam):
    # shape (M, N, K) -> (J, K)
    gamma_mat_cpx = b_operator(boundary, normvec, v_src, k)
    gamma_mat = np.concatenate([np.concatenate([np.real(gamma_mat_cpx), -np.imag(gamma_mat_cpx)], axis=1),
                                np.concatenate([np.imag(gamma_mat_cpx), np.real(gamma_mat_cpx)], axis=1)],
                               axis=0)
    lasso_fitter = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-8)
    scale = np.sqrt(len(gamma_mat))  # rescaling factor for sklearn convention
    lasso_fitter.fit(scale * gamma_mat, scale * target)
    nb_src = len(v_src)
    ampl = lasso_fitter.coef_[:nb_src] + 1j*lasso_fitter.coef_[nb_src:]
    res = y - np.dot(gamma_mat_cpx, ampl)
    return ampl, res


def greedy_mfs_2d(poly, boundary, normvec, src_loc, b_operator, xmin, xmax, ymin, ymax, ngrid, k=2., niter=400,
                  tol=1e-3, nkeep=1, min_dist=0., lam=1e-3, init=None):
    if type(poly) is not Polygon:
        poly = Polygon(poly)

    src_loc = np.reshape(src_loc, (1, 2))

    # cartesian intialization grid
    gridx, gridy = np.meshgrid(np.linspace(xmin, xmax, ngrid), np.linspace(ymin, ymax, ngrid))
    grid = np.stack([gridx.flatten(), gridy.flatten()], axis=1)
    y = -np.reshape(b_operator(boundary, normvec, src_loc, k), -1)  # shape (nboundary,)
    target = np.concatenate([np.real(y), np.imag(y)]).reshape(-1, 1)  # real vector for lasso fitting

    if init is None:
        res, nsrc = y, 0
        niter = len(boundary)
    else:
        ampl, res = fit_weights(y, target, boundary, normvec, b_operator, k, init, lam)
        v_src, nsrc = init, len(init)
        niter = len(boundary) - nsrc

    not_inside = out_polygon(grid, poly, min_dist).reshape(ngrid, ngrid)
    grid_outside = grid[not_inside.flatten(), :]

    for i in range(niter):
        # map gamma at each grid point
        gamma_vec = b_operator(boundary, normvec, grid_outside, k)  # shape (nboundary, ninside**2)
        eta_val = np.sum(np.real(np.conj(gamma_vec)*res[:, np.newaxis]), axis=0)  # value of eta at every grid point
        gridval = np.zeros(ngrid**2)
        gridval[not_inside.flatten()] = eta_val
        if i == niter-1:
            plt.imshow(gridval.reshape(ngrid, ngrid), extent=[xmin, xmax, ymin, ymax])
            plt.plot(v_src[:, 0], v_src[:, 1], '.', label='virtual boundary', color='red')
            plt.colorbar()
            plt.show()

        if nsrc == 0:
            v_src = grid[not_inside.flatten()][np.argmax(eta_val)].reshape([1, -1])
        else:
            v_src = np.concatenate([v_src, grid[not_inside.flatten()][np.argmax(eta_val)].reshape([1, -1])], axis=0)

        # todo : gridless search
        # p = mp.Pool(8)  # todo: get number of cores
        # optivals = p.map(opti_eta, bestvals)
        # opti_ind, opti_val = 0, 0.
        # for j, el in enumerate(optivals):
        #     if el.fun > opti_val:
        #         opti_ind, opti_val = j, el.fun
        # v_src = np.concatenate([v_src, optivals[opti_ind].x], axis=0)
        # optimize amplitudes

        ampl, res = fit_weights(y, target, boundary, normvec, b_operator, k, v_src, lam)
        nsrc += 1
        print("residue : ", np.linalg.norm(res))

    return v_src, ampl
