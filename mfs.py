import numpy as np
from scipy.special import hankel1, jv
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import gmres
from utils import Polygon, ExtrudedPolygon, Cuboid
import matplotlib.pyplot as plt
#from mayavi import mlab


def fundamental_solution(points, sources, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    return (1j/4) * hankel1(0, k*np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=-1))


def fundamental_solution_derivative(dist, k):
    """Compute the derivative of the fundamental solution at the points given the distances,
    return a matrix of shape npoints, nsources."""
    return -(1j/4) * k * hankel1(1, k*dist)


def fundamental_solution3d(points, sources, k):
    """Compute the fundamental solution at the points given the sources, return a matrix of shape npoints, nsources."""
    norms = np.linalg.norm(points[:, np.newaxis, :] - sources[np.newaxis, :, :], axis=-1)
    return np.exp(-1j*k * norms) / norms / (4*np.pi)


def fundamental_solution_derivative3d(dist, k):
    """Compute the derivative of the fundamental solution at the points given the distances,
    return a matrix of shape npoints, nsources."""
    return -np.exp(-1j*k * dist) / dist * (1j*k + 1 / dist) / (4*np.pi)


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


def particular_solution_gradient3d(dist, theta, k, mu, polar_offset):
    return np.concatenate([particular_solution_gradient(dist, theta, k, mu, polar_offset),
                           np.zeros((dist.shape[0], dist.shape[1], 1))], axis=-1)


def neumann_operator(points, normvec, alpha, sourcesv, k, return_diffs=False,
                     fund_der_wrap=fundamental_solution_derivative, individual_contr=False):
    """ Computes the normal derivative of the solution given by the virtual sources sourcesv with coefficients alpha at
    the points 'points' with normal vector normvec. Return the contribution from the virtual sources.
    """
    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    fund_der = fund_der_wrap(norms, k)

    vcontr = (np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) /  # (np, 2)*(np, npv, 2) -> (npv,)
              norms * fund_der*alpha.reshape(1, -1))

    if not individual_contr:
        vcontr = np.sum(vcontr, axis=1)

    if return_diffs:
        return vcontr, diffs, norms
    else:
        return vcontr


def robin_operator(points, normvec, alpha, sourcesv, k, beta, fund_wrap, fund_der_wrap, individual_contr=False):
    if isinstance(beta, (int, float, complex, np.int64, np.float64, np.complex128)):
        beta = np.ones(len(points), dtype=np.complex128)*beta
    else:
        if len(beta) != len(points):
            beta = np.ones(len(points), dtype=np.complex128)*beta[0]

    diffs = points[:, np.newaxis, :] - sourcesv[np.newaxis, :, :]  # (np, npv, 2)
    norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)
    vcontr = ((np.sum(normvec[:, np.newaxis, :] * diffs, axis=-1) / norms *  # (np, 2)*(np, npv, 2) -> (npv,)
               fund_der_wrap(norms, k) + fund_wrap(points, sourcesv, k) *
               beta.reshape((-1, 1)) * 1j * k) * alpha.reshape(1, -1))

    if not individual_contr:
        vcontr = np.sum(vcontr, axis=1)

    return vcontr


def dirichlet_operator(points, _, alpha, sourcesv, k, fund_wrap, individual_contr=False):
    res = fund_wrap(points, sourcesv, k) * alpha.reshape(1, -1)
    if not individual_contr:
        res = np.sum(res, axis=1)
    return res


class SampledBoundary:
    """Class to sample a boundary and compute the normal vector at the sampled points."""
    def __init__(self, poly, sample_param, dx=0., distribution='uniform', min_dist=0.01, sample_type="separation"):
        if sample_type == 'per_wavelength':  # sample nsrc points per wavelength where sample_param=(nsrc, k)
            src_sep = 2*np.pi/sample_param[1]/(sample_param[0] - 1)
            self.bound, self.normvec, self.bound_ids = poly.sample(src_sep, distribution=distribution,
                                                                   min_dist=min_dist, sample_type="separation")
        elif sample_type == 'no_resample':
            self.bound, self.normvec, self.bound_ids = sample_param
        else:
            self.bound, self.normvec, self.bound_ids = poly.sample(sample_param, distribution=distribution,
                                                                   min_dist=min_dist, sample_type=sample_type)
        self.shift(dx)

    def shift(self, dx):
        self.bound += dx*self.normvec
        return self

    def __len__(self):
        return len(self.bound)


class HelmholtzMFS:
    def __init__(self, room_bound: SampledBoundary, virtual_bound: SampledBoundary, pos_src: np.ndarray, poly: Polygon,
                 cond='neumann', beta=None, add_fun=0, ampl=None, src_ampl=None):
        """
        Args:
            pos_src: position of the impulsive sources (shape (nsrc, d))
            poly: Polygon or ExtrudedPolygon instance
            cond: boundary condition, 'dirichlet', 'neumann' or 'robin'
            beta: list of impedance parameters used in the robin boundary condition. Should be a vector containing as
                many elements as there are walls
            add_fun: additional particular solutions to add to the solution
            ampl: preset amplitudes for the virtual sources/particular solutions in case of precomputed solutions
            src_ampl: amplitudes of the impulsive sources (shape (pos_src.shape[0],))
        """

        if isinstance(poly, ExtrudedPolygon) or isinstance(poly, Cuboid):
            self.d, self.fund_wrap, self.fund_der_wrap = 3, fundamental_solution3d, fundamental_solution_derivative3d
            self.particular_wrap, self.particular_der_wrap = particular_solution, particular_solution_gradient3d
        elif isinstance(poly, Polygon):
            self.d, self.fund_wrap, self.fund_der_wrap = 2, fundamental_solution, fundamental_solution_derivative
            self.particular_wrap, self.particular_der_wrap = particular_solution, particular_solution_gradient
        else:
            raise ValueError("poly must be a Polygon or ExtrudedPolygon instance")

        self.pos_src = pos_src.reshape([-1, self.d])
        if src_ampl is None:
            self.src_ampl = np.ones(len(pos_src))
        else:
            self.src_ampl = src_ampl
        self.poly = poly
        self.cond = cond
        self.add_fun = add_fun
        self.room_bound, self.virtual_bound = room_bound, virtual_bound
        self.nbound = len(self.room_bound)
        self.nsrc = len(self.virtual_bound)

        if beta is None:  # default null value for beta
            self.beta = np.zeros(self.poly.nedges + 2*(self.d - 2))
        else:
            # if beta is given, it must be a vector of length nedges
            assert len(beta) == self.poly.nedges + 2*(self.d - 2), "beta must be a vector of length nedges if d=2, " \
                                                                      "nedges + 2 if d=3"
            self.beta = np.asarray(beta)
        self.beta_is_callable = callable(beta[0])


        if ampl is not None:
            self.ampl = ampl
        else:
            self.ampl = np.zeros(self.nsrc, dtype=np.complex128)

        self.factor = np.pi / self.poly.angles[np.newaxis, :]  # angular factor for particular solution

    def mfs_fit(self, k, solver='lstsq', stat="cond"):

        # assemble the system matrix and rhs
        A = np.zeros((self.nbound, self.nsrc + self.add_fun*self.poly.nedges), dtype=complex)
        b = np.zeros((self.nbound, 1), dtype=complex)
        self.ampl = np.ones(self.nsrc + self.add_fun*self.poly.nedges, dtype=complex)
        A[:, :] = self.boundary_operator(self.room_bound.bound, self.room_bound.bound_ids, k, use_src=False,
                                         individual_contr=True)

        b[:] = - np.reshape(self.boundary_operator(self.room_bound.bound, self.room_bound.bound_ids,
                                                   k, use_src=True), [-1, 1])

        if solver == "direct":
            res = np.linalg.solve(A, b)
        elif solver == 'lstsq':
            try:
                res = np.linalg.lstsq(A, b, rcond=-1)[0]
            except:
                print("solve error, freq:", 343*k/np.pi/2)
                exit(1)
                
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

    def evaluate_sol(self, points, k):
        res = np.sum(self.fund_wrap(points, self.virtual_bound.bound, k) * self.ampl[:self.nsrc].reshape(1, -1), axis=1)
        res += np.sum(self.fund_wrap(points, self.pos_src*self.src_ampl.reshape(1, -1), k), axis=1)

        if self.add_fun > 0:
            angles, diffs, norms = self.poly.compute_angles(points)
            factor = self.factor
            indstart, indstop = self.nsrc, self.nsrc + self.poly.nedges
            for t in range(self.add_fun):
                res += np.sum(self.particular_wrap(norms, angles, k, t*factor) *
                              self.ampl[indstart:indstop].reshape(1, -1), axis=1)

                indstart, indstop = indstop, indstop + self.poly.nedges

        return res

    def boundary_operator(self, points, bound_ids, k, use_src=False, individual_contr=False):
        normals = self.poly.get_normal(bound_ids)

        if use_src:
            pos = self.pos_src
            alpha = self.src_ampl
        else:
            alpha = self.ampl
            pos = self.virtual_bound.bound

        if self.cond == 'neumann':
            res = neumann_operator(points, normals, alpha[:self.nsrc], pos,
                                   k, fund_der_wrap=self.fund_der_wrap, individual_contr=individual_contr)
        elif self.cond == 'dirichlet':
            res = dirichlet_operator(points, None, alpha[:self.nsrc], pos,
                                     k, fund_wrap=self.fund_wrap, individual_contr=individual_contr)
        elif self.cond == 'robin':
            if self.beta_is_callable:
                beta_tmp = np.array([beta_fun(k) for beta_fun in self.beta])
            else:
                beta_tmp = self.beta
                
            beta = beta_tmp[bound_ids]

            res = robin_operator(points, normals, alpha[:self.nsrc], pos, k, beta, fund_wrap=self.fund_wrap,
                                 fund_der_wrap=self.fund_der_wrap, individual_contr=individual_contr)

        else:
            raise ValueError("Unknown boundary condition")

        if self.add_fun > 0:
            angles, diffs, norms = self.poly.compute_angles(points)

            indstart, indstop = self.nsrc, self.nsrc + self.poly.nedges
            if not use_src:
                for t in range(self.add_fun):
                    grad = self.particular_der_wrap(norms, angles, k, t * self.factor,
                                                    self.poly.polar_angles)  # (np, nvertices, 2)
                    if self.cond == 'neumann':
                        add_contr = (np.sum(grad * normals[:, np.newaxis, :], axis=-1) *
                                     self.ampl[indstart:indstop].reshape(1, -1))
                    elif self.cond == 'robin':
                        add_contr = ((np.sum(grad * normals[:, np.newaxis, :], axis=-1) + beta.reshape(-1, 1) *
                                      k*1j*self.particular_wrap(norms, angles, k, t*self.factor)) *
                                     self.ampl[indstart:indstop].reshape(1, -1))
                    elif self.cond == 'dirichlet':
                        add_contr = (self.particular_wrap(norms, angles, k, t*self.factor) *
                                     self.ampl[indstart:indstop].reshape(1, -1))
                    else:
                        raise ValueError("Unknown boundary condition")

                    if not individual_contr:
                        res += np.sum(add_contr, axis=1)
                    else:
                        res = np.concatenate([res, add_contr], axis=1)
                    indstart, indstop = indstop, indstop + self.poly.nedges
        return res

    def plot_bound_values(self, nplot, k, show=True, name="", plot3d=False, scale=0.95):
        if self.d == 2:
            mfs_val, src_val, edges, plot_nodes = [], [], [t * nplot for t in range(1, self.poly.nedges)], []
            for j in range(self.poly.nedges):
                plot_nodes.append(self.poly.sample_edge(nplot, j, distribution="uniform"))
                mfs_val.append(self.boundary_operator(plot_nodes[-1], j*np.ones(nplot, dtype=int), k))
                src_val.append(-self.boundary_operator(plot_nodes[-1], j*np.ones(nplot, dtype=int), k, use_src=True))

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
            ax3.vlines(np.where(np.diff(self.virtual_bound.bound_ids) != 0)[0],
                       np.minimum(np.min(self.ampl.real), np.min(self.ampl.imag)),
                       np.maximum(np.max(self.ampl.real), np.max(self.ampl.imag)), colors="k", linestyles="dashed")
            plt.legend()
            if show:
                plt.show()
            else:  # save the figure
                plt.savefig("data/fig/Neumann_edge_{}_nsrc={}.png".format(name, self.nsrc))

        else:  # plot boundary values in 3d, nplot is the separation for the resampled boundary
            boundplot, normplot, ids, tri = self.poly.sample(nplot, min_dist=0.005, return_triangulation=True)
            print("Plotting {} points on the boundary".format(len(boundplot)))
            mfs_val = self.boundary_operator(boundplot, ids, k)
            src_val = -self.boundary_operator(boundplot, ids, k, use_src=True)
            error_real = np.sqrt(np.sum((np.real(mfs_val) - np.real(src_val)) ** 2))/np.linalg.norm(src_val.real)
            error_imag = np.sqrt(np.sum((np.imag(mfs_val) - np.imag(src_val)) ** 2))/np.linalg.norm(src_val.imag)
            print("Relative error on the real part: {:3f}".format(error_real))
            print("Relative error on the imaginary part: {:3f}".format(error_imag))
            mlab.figure()
            min_val, max_val = np.min(np.real(src_val))*scale, np.max(np.real(src_val))*scale
            for i in range(self.poly.nedges+2):
                plot_nodes = boundplot[ids == i]
                triz = tri[i]
                mesh = mlab.triangular_mesh(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], triz,
                                            scalars=np.real(src_val[ids == i]), vmin=min_val, vmax=max_val,
                                            representation='surface')
            cbar = mlab.colorbar(orientation='vertical')
            cbar.data_range = (min_val, max_val)
            mlab.figure()
            for i in range(self.poly.nedges+2):
                plot_nodes = boundplot[ids == i]
                triz = tri[i]
                mesh = mlab.triangular_mesh(plot_nodes[:, 0], plot_nodes[:, 1], plot_nodes[:, 2], triz,
                                            scalars=np.real(mfs_val[ids == i]), vmin=min_val, vmax=max_val,
                                            representation='surface')
            cbar2 = mlab.colorbar(orientation='vertical')
            cbar2.data_range = (min_val, max_val)
            mlab.show()

    def plot_boundary(self, show=True):
        """Plot the boundary of the polygon and the sources"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title("Boundary of the polygon")
        ax.set_aspect('equal')
        ax.scatter(self.room_bound.bound[:, 0], self.room_bound.bound[:, 1], c='k', label="boundary")
        ax.scatter(self.virtual_bound.bound[:, 0], self.virtual_bound.bound[:, 1], c='r', label="virtual boundary")
        plt.legend()
        if show:
            plt.show()

    def update_sources(self, new_vbound):
        self.virtual_bound = new_vbound
        self.nsrc = len(self.virtual_bound)

    def update_source_term(self, new_pos):
        self.pos_src = new_pos

    def update_boundary(self, new_bound):
        self.room_bound = new_bound
        self.nbound = len(self.room_bound)
