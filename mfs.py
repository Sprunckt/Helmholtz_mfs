import numpy as np
from scipy.special import hankel1, jv
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import gmres
from utils import Polygon
from scipy.optimize import minimize
from scipy.integrate import simpson
import time
import matplotlib.pyplot as plt


def fundamental_solution(points, sources, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    return (1j/4) * hankel1(0, k*np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=-1))


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


def compute_sol(alpha, points, sourcesv, k, source=None, decouple=False, individual_contr=False):
    """Compute the solution at the points given the virtual sources and the associated coefficients.
    If decouple is True, alpha is supposed tobe of shape 2n, the first n coefficients correspond to the real part of the
    fundamental solutions, the last n to the imaginary part."""
    ncoeff = len(alpha)
    fund_sol = fundamental_solution(points, sourcesv, k)

    if decouple and ncoeff > 1:
        res = (np.reshape(alpha[:ncoeff//2], -1)[np.newaxis, :] * np.real(fund_sol) +
               np.reshape(alpha[ncoeff//2:], -1)[np.newaxis, :] * np.imag(fund_sol))

    else:
        res = np.reshape(alpha, -1)[np.newaxis, :] * fund_sol
    if not individual_contr:
        res = np.sum(res, axis=1)
    if source is not None:
        return res + np.reshape(fundamental_solution(points, source, k), -1)
    else:
        return res


def compute_exact_sol(points, image_src, k, weights=None):
    """Compute the exact solution at the points given the source position."""
    if weights is None:
        weights = np.ones(len(image_src))
    return np.sum(weights[np.newaxis, :] * fundamental_solution(points, image_src, k), axis=1)


def solve(beta, normvec, bound, boundv, source, k, source_type='dirac', cond='dirichlet', method='direct', offset=None,
          return_mat=False, decouple=False):
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

        # (np, 2)*(np, npv, 2)->(np, npv)
        scalar_prod = np.sum(normvec[:indstop, np.newaxis, :] * diffs, axis=-1) / norms
        A[:indstop, :] = scalar_prod * fundamental_solution_derivative(norms, k)
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
        A[-1, :] = np.ones(nsrc)
        b[-1] = 1.

    if decouple:
        A = np.concatenate([np.real(A), np.imag(A)], axis=1)

    if offset is not None:
        b += offset.reshape(-1, 1)

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


def neumann_operator(points, normvec, alpha, sourcesv, k, decouple=False, return_diffs=False,
                     individual_contr=False):
    """ Computes the normal derivative of the solution given by the virtual sources sourcesv with coefficients alpha at
    the points 'points' with normal vector normvec. Return the contribution from the virtual sources.
    """
    ncoeff = len(alpha)
    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    fund_der = fundamental_solution_derivative(norms, k)

    if decouple and ncoeff > 1:
        vcontr = np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) / norms * (
                np.real(fund_der) * alpha[:ncoeff // 2].reshape(1, -1) +
                np.imag(fund_der) * alpha[ncoeff // 2:].reshape(1, -1))
        if not individual_contr:
            vcontr = np.sum(vcontr, axis=1)

    else:
        vcontr = (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, npv, 2) -> (npv,)
                  norms * fund_der*alpha.reshape(1, -1))

        if not individual_contr:
            vcontr = np.sum(vcontr, axis=1)

    if return_diffs:
        return vcontr, diffs, norms
    else:
        return vcontr


def robin_operator(points, normvec, alpha, sourcesv, k, beta, individual_contr=False):
    if isinstance(beta, (int, float, complex, np.int64, np.float64, np.complex128)):
        beta = np.ones(len(points), dtype=np.complex128)*beta
    else:
        beta = np.ones(len(points), dtype=np.complex128)*beta[0]

    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    vcontr = ((np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) / norms * # (np, 2)*(np, npv, 2) -> (npv,)
               fundamental_solution_derivative(norms, k) + fundamental_solution(points, sourcesv, k) *
               beta.reshape((-1, 1)) * 1j * k) * alpha.reshape(1, -1))

    if not individual_contr:
        vcontr = np.sum(vcontr, axis=1)

    return vcontr


def dirichlet_operator(points, _, alpha, sourcesv, k, individual_contr=False):
    """ Computes the solution given by the virtual sources sourcesv with coefficients alpha at
    the points 'points' with normal vector normvec. Return the contribution from the virtual sources.
    """

    return compute_sol(alpha, points, sourcesv, k, individual_contr=individual_contr)


class SlidingMFS:
    def __init__(self, nbound: int, nsrc: int, nobs: int, pos_src: np.ndarray, poly: Polygon,
                 k: float, dx: float = None, cond='neumann', lam=0., sample_kwargs=None, beta=None, src_nb='total',
                 add_fun=0):
        self.dx = dx
        self.pos_src = pos_src
        self.poly = poly
        self.k = k
        self.cond = cond
        self.nbound, self.nobs = nbound, nobs
        self.lam = lam
        if sample_kwargs is None:
            sample_kwargs = {"distribution": 'uniform', "min_dist": 0.05}
        self.bound, self.normvec, self.bound_ids, self.params = poly.sample(nbound, **sample_kwargs,
                                                                            return_ids=True, return_nodes=True)
        self.bound_obs, self.normvec_obs = poly.sample(nobs, distribution='cheby')

        sample_kwargs["min_dist"], sample_kwargs["distribution"] = 0., "uniform"
        if src_nb == 'total':
            boundv, self.normvecv, self.boundv_ids, self.paramsv = poly.sample(nsrc, **sample_kwargs,
                                                                               return_ids=True, return_nodes=True)
        elif src_nb == 'per_wavelength':  # sample nsrc points per wavelength
            src_sep = 2*np.pi/k/(nsrc - 1)
            boundv, self.normvecv, self.boundv_ids, self.paramsv = poly.sample(src_sep, **sample_kwargs, sample_type="sep",
                                                                               return_ids=True, return_nodes=True)
        else:
            raise ValueError("src_nb must be 'total' or 'per_wavelength'")
        self.boundv = boundv + dx*self.normvecv

        self.nsrc = len(self.boundv)

        # self.boundv = poly.sample_rounded(self.nsrc, dx, factor=1, one_per_corner=True)

        if beta is None:
            self.beta = np.zeros(self.nbound)
        else:
            self.beta = beta*np.ones(self.nbound)

        self.ampl = np.zeros(self.nsrc, dtype=np.complex128)

        self.add_fun = add_fun
        self.factor = np.pi / self.poly.angles[np.newaxis, :]  # angular factor for particular solution

    def gram_fit(self, verbose=True, lam=None, decouple=False, lam_gradient=None):
        """Compute scalar products between fundamental solutions, use the boundary points as quadrature points"""
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
            scontr = fundamental_solution(self.bound, self.pos_src.reshape([1, -1]), self.k).squeeze()

        if decouple:
            vcontr = np.concatenate((vcontr.real, vcontr.imag), axis=1)
            # self.poly.bound_ids = np.concatenate([self.poly.bound_ids, self.poly.bound_ids])
            nsrc = 2*self.nsrc
            A = np.zeros((nsrc, nsrc), dtype=np.float)
            b = np.zeros(nsrc, dtype=np.complex)

        else:
            nsrc = self.nsrc
            A = np.zeros((nsrc, nsrc), dtype=np.complex128)
            b = np.zeros((nsrc, 1), dtype=np.complex128)

        for i in range(nsrc):
            for j in range(i, nsrc):
                # compute L2 scalar product between green functions
                prod = vcontr[:, i].conj() * vcontr[:, j]  # (npo,)
                # apply quadrature rule (simpson rule) over each edge
                for t in range(self.poly.nedges):
                    ind_edge = self.bound_ids == t
                    A[i, j] += simpson(prod[ind_edge], self.params[ind_edge], axis=0)*self.poly.lengths[t]
                A[j, i] = A[i, j].conj()
            for t in range(self.poly.nedges):
                ind_edge = self.bound_ids == t

                b[i] += simpson(vcontr[ind_edge, i]*scontr[ind_edge].conj(),
                                self.params[ind_edge], axis=0)*self.poly.lengths[t]

        t2 = time.time()
        if verbose:
            print("Gram matrix computation took {:.3f} s".format(t2 - t1))

        if decouple:
            real_part = np.linalg.lstsq(A, b.real, rcond=-1)[0]  # fitting the real part of the source contribution
            imag_part = np.linalg.lstsq(A, b.imag, rcond=-1)[0]

            self.ampl = - real_part + 1j * imag_part
            cond = np.linalg.cond(A)
        else:
            real_matrix = np.concatenate([np.concatenate([np.real(A), -np.imag(A)], axis=1),
                                          np.concatenate([np.imag(A), np.real(A)], axis=1)], axis=0)

            if lam is not None:
                penal = lam*np.eye(2*self.nsrc)
                real_matrix += penal
            real_rhs = np.concatenate([-np.real(b), np.imag(b)])
            cond = np.linalg.cond(real_matrix)
            self.ampl = np.linalg.lstsq(real_matrix, real_rhs, rcond=-1)[0]
            self.ampl = self.ampl[:self.nsrc] + 1j * self.ampl[self.nsrc:]

        t3 = time.time()
        if verbose:
            print("Solving took {:.3f} s".format(t3 - t2))

        return self.ampl, cond

    def mfs_fit(self, solver='lstsq', stat="cond"):

        # assemble the system matrix and rhs
        A = np.zeros((self.nbound, self.nsrc + self.add_fun*self.poly.nedges), dtype=complex)
        b = np.zeros((self.nbound, 1), dtype=complex)
        self.ampl = np.ones(self.nsrc + self.add_fun*self.poly.nedges, dtype=complex)
        A[:, :] = self.boundary_operator(self.bound, self.bound_ids, use_src=False, individual_contr=True)

        b[:] = - np.reshape(self.boundary_operator(self.bound, self.bound_ids, use_src=True), [-1, 1])

        if solver == "direct":
            res = np.linalg.solve(A, b)
        elif solver == 'lstsq':
            res = np.linalg.lstsq(A, b, rcond=-1)[0]
        elif solver == "tsvd":
            res = np.linalg.pinv(A) @ b
        elif solver == "gmres":
            res = gmres(A, b, tol=1e-8)[0]
        elif solver == "ridge":
            lasso_fitter = Ridge(alpha=1e-3, fit_intercept=False, tol=1e-8)
            Ac1 = np.concatenate([np.real(A), -np.imag(A)], axis=1)
            Ac2 = np.concatenate([np.imag(A), np.real(A)], axis=1)
            A = np.concatenate([Ac1, Ac2], axis=0)
            b = np.concatenate([np.real(b), np.imag(b)], axis=0)
            scale = np.sqrt(len(A))  # rescaling factor for sklearn convention
            lasso_fitter.fit(scale * A, scale * b)
            res = lasso_fitter.coef_.flatten()
            res = res[:self.nsrc] + 1j * res[self.nsrc:]
        else:
            raise ValueError("method must be 'direct', 'tsvd', 'lstsq', 'ridge' or 'gmres'")

        self.ampl = res.squeeze()
        if stat == "cond":
            stat = np.linalg.cond(A)
        elif stat == "det":
            stat = np.linalg.det(A)
        elif stat == "sing":  # smallest singular value
            stat = np.linalg.svd(A, compute_uv=False)[-1]
        else:
            raise ValueError("Unknown stat")
        return self.ampl, stat

    def evaluate_sol(self, points):
        res = np.sum(fundamental_solution(points, self.boundv, self.k) * self.ampl[:self.nsrc].reshape(1, -1), axis=1)
        res += np.reshape(fundamental_solution(points, self.pos_src, self.k), -1)

        angles, diffs, norms = self.poly.compute_angles(points)
        factor = self.factor
        indstart, indstop = self.nsrc, self.nsrc + self.poly.nedges
        for t in range(self.add_fun):
            res += np.sum(particular_solution(norms, angles, self.k, t*factor) *
                          self.ampl[indstart:indstop].reshape(1, -1), axis=1)

            indstart, indstop = indstop, indstop + self.poly.nedges

        return res

    def boundary_operator(self, points, bound_ids, use_src=False, individual_contr=False):
        normals = self.poly.normals[bound_ids]
        if use_src:
            pos = self.pos_src.reshape([1, -1])
            alpha = np.ones(1)
        else:
            alpha = self.ampl
            pos = self.boundv
        if self.cond == 'neumann':
            res = neumann_operator(points, normals, alpha[:self.nsrc], pos, self.k, individual_contr=individual_contr)

            if self.add_fun > 0:
                angles, diffs, norms = self.poly.compute_angles(points)

                indstart, indstop = self.nsrc, self.nsrc + self.poly.nedges
                if not use_src:
                    for t in range(self.add_fun):
                        grad = particular_solution_gradient(norms, angles, self.k, t*self.factor,
                                                            self.poly.polar_angles)  # (np, nvertices, 2)

                        add_contr = (np.sum(grad * normals[:, np.newaxis, :], axis=-1) *
                                     self.ampl[indstart:indstop].reshape(1, -1))
                        if not individual_contr:
                            res += np.sum(add_contr, axis=1)
                        else:
                            res = np.concatenate([res, add_contr], axis=1)
                        indstart, indstop = indstop, indstop + self.poly.nedges
            return res
        elif self.cond == 'dirichlet':
            return dirichlet_operator(points, None, alpha, pos, self.k, individual_contr=individual_contr)
        elif self.cond == 'robin':
            return robin_operator(points, normals, alpha, pos, self.k, self.beta, individual_contr=individual_contr)
        else:
            raise ValueError("Unknown boundary condition")

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

    def plot_bound_values(self, nplot, show=True, name="", plot3d=False):
        mfs_val, src_val, edges, plot_nodes = [], [], [t * nplot for t in range(1, self.poly.nedges)], []
        for j in range(self.poly.nedges):
            plot_nodes.append(self.poly.sample_edge(nplot, j, distribution="uniform"))
            mfs_val.append(self.boundary_operator(plot_nodes[-1], j*np.ones(nplot, dtype=int)))
            src_val.append(-self.boundary_operator(plot_nodes[-1], j*np.ones(nplot, dtype=int), use_src=True))

        mfs_val, src_val = np.concatenate(mfs_val), np.concatenate(src_val)
        error_real = np.sqrt(np.mean(np.abs(np.real(mfs_val) - np.real(src_val)) ** 2))
        error_imag = np.sqrt(np.mean(np.abs(np.imag(mfs_val) - np.imag(src_val)) ** 2))

        if plot3d:
            plot_nodes = np.concatenate(plot_nodes, axis=0)
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_title("Real part, nsrc={}, nb={}, error={:3f}".format(self.nsrc, self.nbound, error_real))
            # plot the boundary values above the polygon
            ax1.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.real(mfs_val), label="MFS")
            ax1.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.real(src_val), '--', label="Source contribution")

            ax2 = fig.add_subplot(222, projection='3d')
            ax2.set_title("Imaginary part, error={:3f}".format(error_imag))
            ax2.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.imag(mfs_val), label="MFS")
            ax2.plot(plot_nodes[:, 0], plot_nodes[:, 1], np.imag(src_val), '--', label="Source contribution")

            for ax in [ax1, ax2]:  # plot the polygon on the z=0 plane
                ax.plot(self.poly.vertices[:, 0], self.poly.vertices[:, 1], np.zeros(self.poly.nedges), 'k')
                ax.plot([self.poly.vertices[0, 0], self.poly.vertices[-1, 0]],
                        [self.poly.vertices[0, 1], self.poly.vertices[-1, 1]], [0, 0.], 'k')
            # set equal aspect ratio for the 3d plot
            max_val = np.max(np.abs(plot_nodes), axis=0)
            min_val = np.min(np.abs(plot_nodes), axis=0)
            ax1.set_xlim([min_val[0], max_val[0]]), ax1.set_ylim([min_val[1], max_val[1]])
            ax2.set_xlim([min_val[0], max_val[0]]), ax2.set_ylim([min_val[1], max_val[1]])
            plt.legend()

        else:
            fig, ax = plt.subplots(2, 2, figsize=(20, 6))
            ax[0, 0].set_ylim([np.min(src_val.real), np.max(src_val.real)])
            ax[0, 1].set_ylim([np.min(src_val.imag), np.max(src_val.imag)])

            ax[0, 0].set_title("Real part, nsrc={}, mean error={:3f}".format(self.nsrc, error_real))
            ax[0, 1].set_title("Imaginary part, mean error={:3f}".format(error_imag))
            ax[0, 0].plot(np.real(mfs_val), label="MFS")
            ax[0, 0].plot(np.real(src_val), '--', label="Source contribution")
            ax[0, 1].plot(np.imag(mfs_val), label="MFS")
            ax[0, 1].plot(np.imag(src_val), '--', label="Source contribution")

            # add horizontal line at 0
            ax[0, 0].plot([0, len(mfs_val)], [0, 0], 'k')
            ax[0, 1].plot([0, len(mfs_val)], [0, 0], 'k')
            # add vertical bars at each edge change
            ax[0, 0].vlines(edges, np.min(src_val.real), np.max(src_val.real), colors="k", linestyles="dashed")
            ax[0, 1].vlines(edges, np.min(src_val.imag), np.max(src_val.imag), colors="k", linestyles="dashed")
            plt.legend()
            plt.tight_layout()

        # add a subplot under both plots to show the coefficients
        ax3 = fig.add_subplot(212)
        ax3.set_title("Coefficients")
        ax3.semilogy(np.abs(self.ampl.real), '.', label="real part")
        ax3.semilogy(np.abs(self.ampl.imag), '.', label="imaginary part")
        # plot a vline at each edge change (given by value changes in coeffs_id)
        ax3.vlines(np.where(np.diff(self.boundv_ids) != 0)[0], np.minimum(np.min(self.ampl.real), np.min(self.ampl.imag)),
                   np.maximum(np.max(self.ampl.real), np.max(self.ampl.imag)), colors="k", linestyles="dashed")
        plt.legend()
        if show:
            plt.show()
        else:  # save the figure
            plt.savefig("data/fig/Neumann_edge_{}_nsrc={}.png".format(name, self.nsrc))
