import numpy as np
from mfs import (SampledBoundary, particular_solution_gradient, robin_operator, particular_solution,
                 fundamental_solution, fundamental_solution_derivative,particular_solution_tangential_gradient)
from utils import Polygon, c
import multiprocessing as mp
import time
from functools import partial
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, vertices: np.ndarray, fs: float, nfreq: int, dx: float, pos_src: np.ndarray, mic_pos: np.ndarray,
                 observations=None, add_src=15, nsrc_schedule=None, min_dist=0.01, error_thresh=0.05,
                 sep_red=0.8, src_ampl=None, beta=0.1j, origin=None, vmin_dist=0):
        """
        Simulation class for Helmholtz equation: simulate a source at pos_src in a polygonal domain with boundary
        conditions at different frequencies. Can generate time-domain signals by applying an inverse Fourier transform.
        Args:
            - vertices: vertices of the polygonal domain
            - fs: sampling frequency
            - nfreq: number of frequencies
            - dx: distance between real and virtual boundaries
            - pos_src: position of the source
            - add_src: number of additional sources in the enriched MFS basis
            - nsrc_schedule: strategy to sample the virtual boundary. Should contain a key "sampling_method" which can
        be either "separation" or "per_wavelength". If the method is "per_wavelength", the key "nsrc" should contain the
         number of sources per wavelength at the maximal frequency. If the method is "separation",
         the distance between sources should be given using the key "sep".
            - min_dist: minimum distance between boundary points and the corners of the polygon
            - error_thresh: threshold for the boundary error at initialization
        """

        if src_ampl is None:
            src_ampl = np.ones(len(pos_src), dtype=np.complex128)
        else:
            assert src_ampl.shape == (len(pos_src),), "src_ampl should have shape (len(pos_src),)"
        self.src_ampl = src_ampl
        self.src_pos = np.reshape(pos_src, [-1, 2])
        self.vmin_dist = vmin_dist

        self.beta, self.beta_adj = beta, -beta
        self.fs, self.dx = fs, dx

        self.poly = Polygon(vertices, origin=origin)
        self.factor = np.pi / self.poly.angles[None, :]  # angular factor for particular solutions

        self.freqs = np.linspace(0, fs, nfreq)
        self.wave_numbers = 2 * np.pi * self.freqs / c
        self.nfreqs = nfreq
        self.min_dist = min_dist
        self.add_src = add_src
        self._to_simulate = np.ones(nfreq, dtype=bool)
        self._to_simulate[0] = False
        self._to_simulate[-1] = False

        if nsrc_schedule is None:
            # sampling strategy : 3 points per wavelength
            nsrc = 10
            nsrc_schedule = dict(method='per_wavelength', nsrc=nsrc)

        if nsrc_schedule["method"] not in ["per_wavelength", "separation"]:
            raise ValueError("sampling method should be either per_wavelength or separation")

        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = c / self.freqs[-1] / (nsrc_schedule["nsrc"] - 1)
        print(nsrc_schedule)
        self.vbound = SampledBoundary(self.poly, nsrc_schedule["sep"],
                                      min_dist=0., sample_type="separation", dx=self.dx)
        self.ampl = np.zeros([self.nfreqs, self.get_nvirt()], dtype=complex)

        # check if there are enough boundary points, take the max number of virtual sources if not
        nbound = self.get_nvirt()*1.1
        # sample boundary
        self.room_bound = SampledBoundary(self.poly, nbound, min_dist=min_dist, sample_type="total")
        print("Number of boundary points: ", self.get_nbound())

        nerror = 1.3 * nbound
        # nodes for computing error
        self.error_bound = SampledBoundary(self.poly, nerror, min_dist=min_dist, sample_type="total")
        print("Number of error points: ", len(self.error_bound))

        print("simulating frequency ", self.freqs[-1])
        rel_error, max_tries = np.inf, 10
        t1 = time.time()
        while rel_error > error_thresh and max_tries > 0:  # increase number of sources until precision threshold is met
            max_tries -= 1
            self.ampl[-1] = self.simulate_freq(-1, verbose=True, simulate_adjoint=False)

            # check error
            rel_error = self.evaluate_error(-1, adjoint=False)

            if rel_error > error_thresh:  # resample with more sources
                nsrc_schedule["sep"] = nsrc_schedule["sep"] * sep_red

                self.vbound = SampledBoundary(self.poly, nsrc_schedule["sep"],
                                              min_dist=self.vmin_dist, sample_type="separation", dx=self.dx)
                self.ampl = np.zeros([self.nfreqs, self.get_nvirt()], dtype=complex)
                print("error {} above threshold, retrying "
                      "with separation {}, {} sources".format(rel_error, nsrc_schedule["sep"],
                                                              len(self.vbound)))

                totnsrc = len(self.vbound) + self.add_src * self.poly.nedges
                if totnsrc > nbound:
                    nbound = totnsrc*1.1
                    print("error above threshold, increasing number of boundary points")
                    self.room_bound = SampledBoundary(self.poly, nbound, min_dist=min_dist, sample_type="total")

                    nerror = 1.3 * nbound
                    self.error_bound = SampledBoundary(self.poly, nerror, min_dist=min_dist, sample_type="total")

            if max_tries == 0:
                print("error threshold not reached, simulation might be inaccurate")

            t2 = time.time()
            print("time {} s".format(t2-t1))
            print("number of sources: {} relative error: {}".format(self.get_nvirt(), rel_error))

        self.nsrc_schedule = nsrc_schedule
        self.ampl_adj = np.zeros([self.nfreqs, self.get_nvirt()], dtype=complex)
        # todo : compute values at each microphone position, compute diffs
        if observations is not None:
            self.obs = observations

        self.mic_pos = mic_pos
        self.src_ampl_adj = np.zeros(len(self.mic_pos), dtype='complex')

    def get_nbound(self):
        return len(self.room_bound)

    def get_nvirt(self):
        return len(self.vbound) + self.add_src * self.poly.nedges

    def update_poly(self, new_vert, bounds=None, check_in=None, convex_proj=False, min_length=None, max_angle=None,
                    increase_factor=1.1):
        success = self.poly.update_vertices(new_vert, bounds=bounds, check_in=check_in, convex_proj=convex_proj,
                                            update_origin=True, min_length=min_length, max_angle=max_angle)
        if success:  # resample polygon after update
            self.vbound = SampledBoundary(self.poly, self.nsrc_schedule["sep"], min_dist=0., dx=self.dx)
            self.room_bound = SampledBoundary(self.poly, sample_param=self.get_nvirt()*increase_factor,
                                              dx=0, min_dist=self.min_dist, sample_type='total')
        else:
            print("Failure to update the polygon")
        return success

    def update_ampl_adj(self):
        self.src_ampl_adj = self.obs - self.compute(self.mic_pos, adj=False)

    def _select_sources(self, ind, adj):
        if adj:
            return self.ampl_adj[ind], self.src_ampl_adj, self.mic_pos
        else:
            return self.ampl[ind], self.src_ampl, self.src_pos

    def compute(self, points, ind=None, adj=False, nworkers=1, verbose=False):
        """Compute the values taken by the state p or the adjoint q at each microphone position."""
        if ind is None:
            ind = np.arange(1, self.nfreqs)

        t1 = time.time()
        if nworkers == 1:
            res = np.full([len(points), len(ind)], np.nan, dtype=np.complex128)
            for k, i in enumerate(ind):
                res[:, k] = self.compute_freq(points, i, adj)
        elif nworkers > 1:
            with mp.get_context('fork').Pool(processes=nworkers) as pool:
                # local function to compute the solution at a given frequency, take only ind as argument
                fun = partial(self.compute_freq, points=points, adj=adj)
                res = np.stack(pool.map(fun, ind), axis=1)
        else:
            print("nworkers arg should be positive")
            exit(1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1))

        return res

    def compute_freq(self, points, ind, adj=False):
        ampl, ampl_src, src_pos = self._select_sources(ind, adj)

        # contribution from virtual sources
        res = np.sum(ampl[None, :len(self.vbound)] *
                     fundamental_solution(np.linalg.norm(points[:, None, :] - self.vbound.bound[None, :, :],
                                                         axis=-1), self.wave_numbers[ind]), axis=1)
        # add contribution from particular solutions
        angles, _, norms = self.poly.compute_angles(points)
        indstart, indstop = len(self.vbound), len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):  # particular solutions contributions
            res += np.sum(particular_solution(norms, angles, self.wave_numbers[ind], t * self.factor) *
                          ampl[indstart:indstop].reshape(1, -1), axis=-1)
            indstart, indstop = indstop, indstop + self.poly.nedges

        # add source contribution
        res += np.sum(ampl_src[None, :]*fundamental_solution(np.linalg.norm(points[:, None, :] - src_pos[None, :, :],
                                                                            axis=-1),
                                                             self.wave_numbers[ind]), axis=-1)

        return res

    def compute_grad_freq(self, points, ind, adj=False):
        """Compute the gradient of the solution at the given points (shape (npoints, d)) at the given frequency"""
        ampl, ampl_src, src_pos = self._select_sources(ind, adj)

        k = self.wave_numbers[ind]
        diffs = points[:, None, :] - self.vbound.bound[None, :, :]  # (np, npv, 2)
        norms = np.linalg.norm(diffs, axis=-1)  # (np, npv)

        grad = np.sum(fundamental_solution_derivative(norms, k)[:, :, None] * diffs
                      / norms[:, :, None] * ampl[None, :len(self.vbound), None], axis=1)  # (np, 2)

        # add source contribution
        diffs = points[:, None, :] - src_pos[None, :, :]  # (np, nsrc, 2)
        norms = np.linalg.norm(diffs, axis=-1)  # (np, nsrc)
        grad += np.sum(fundamental_solution_derivative(norms, k)[:, :, None] * diffs
                       / norms[:, :, None] * ampl_src[None, :, None], axis=1)
        angles, diffs, norms = self.poly.compute_angles(points)
        indstart, indstop = len(self.vbound), len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):
            grad += np.sum(particular_solution_gradient(norms, angles, k, t * self.factor, self.poly.polar_angles) *
                           ampl[None, indstart:indstop, None], axis=1)
            indstart, indstop = indstop, indstop + self.poly.nedges

        return grad  # shape (npoints, nfreq, 2

    def simulate_freq(self, ind, simulate_adjoint=False, verbose=False): # todo: store boundary contributions to avoid recomputing
        """Assemble and solve the MFS system for the frequency of index ind, return the vector of corresponding
        amplitudes (sources and particular solutions)."""
        t1 = time.time()

        # assemble the system matrix and rhs
        self.ampl[ind] = 1  # reset amplitudes to 1 to compute virtual sources contribution
        A = self.boundary_operator(ind, self.room_bound, adjoint=False)

        # source contribution
        b = - np.reshape(self.boundary_operator_source(ind, self.room_bound, adjoint=False), [-1, 1])

        try:
            res = np.linalg.lstsq(A, b, rcond=-1)[0]
        except:
            print("solve error, freq:", 343 * self.wave_numbers[ind] / np.pi / 2, flush=True)
            exit(1)

        t2 = time.time()
        if verbose:
            print("simulating frequency {}; time {} s; {} sources".format(self.freqs[ind], t2-t1, self.get_nvirt()))
        if simulate_adjoint:
            A = self.boundary_operator(ind, self.room_bound, adjoint=True)
            b = - np.reshape(self.boundary_operator_source(ind, self.room_bound, adjoint=True), [-1, 1])
            try:
                res_adj = np.linalg.lstsq(A, b, rcond=-1)[0]
            except:
                print("solve error, freq:", 343 * self.wave_numbers[ind] / np.pi / 2, flush=True)
                exit(1)

            return res.squeeze(), res_adj.squeeze()
        else:
            return res.squeeze()

    def evaluate_error(self, freq_ind, adjoint=False):
        mfs_val = self.boundary_operator(freq_ind, self.error_bound, aggregate=True, adjoint=adjoint)
        src_val = -self.boundary_operator_source(freq_ind, self.error_bound, adjoint=adjoint)

        rel_error = np.sqrt(np.sum((np.abs(mfs_val - src_val)) ** 2)) / np.linalg.norm(src_val)
        return rel_error

    def get_ind(self, fmin, fmax):
        """Return indices from the range of frequencies [fmin, fmax]"""
        return np.where((self.freqs > fmin) & (self.freqs <= fmax))[0]

    def set_freq_ind_to_simulate(self, ind_sim):
        self._to_simulate[:] = False
        self._to_simulate[ind_sim] = True

    def simulate(self, nworkers=1, verbose=True, adjoint=False):
        """Compute the solution coefficients at each frequency using nworkers processes."""
        t1 = time.time()
        ind_comp = np.where(self._to_simulate)[0]
        if nworkers == 1:  # 1 frequency at a time
            if adjoint:
                for ind in ind_comp:
                    self.ampl[ind], self.ampl_adj[ind] = self.simulate_freq(ind, verbose=verbose,
                                                                            simulate_adjoint=True)
            else:
                for ind in ind_comp:
                    self.ampl[ind] = self.simulate_freq(ind, verbose=verbose, simulate_adjoint=False)
        elif nworkers > 1:
            with mp.get_context('fork').Pool(processes=nworkers) as pool:
                partfunc = partial(self.simulate_freq, verbose=verbose, simulate_adjoint=adjoint)
                ampl = pool.map(partfunc, ind_comp)
                if adjoint:
                    for i, ind in enumerate(ind_comp):
                        self.ampl[ind], self.ampl_adj[ind] = ampl[i][0], ampl[i][1]
                else:
                    for i, ind in enumerate(ind_comp):
                        self.ampl[ind] = ampl[i]
        else:
            print("nworkers arg should be positive")
            exit(1)
        if adjoint:  # recompute adjoint sources
            self.src_ampl_adj = self.obs - self.compute(self.mic_pos)

        if verbose:
            print("total simulation time : {} s".format(time.time()-t1))

    def compute_grad(self, compute_points, nworkers=1, verbose=False, freq_ind=None):
        t1 = time.time()
        if freq_ind is None:  # compute all frequencies if not specified
            freq_ind = np.arange(1, self.nfreqs)

        if nworkers == 1:
            res_grad = np.zeros([len(compute_points), len(freq_ind), 2], dtype=np.complex128)

            for k, i in enumerate(freq_ind):
                res_grad[:, k] = self.compute_grad_freq(compute_points, i)
        elif nworkers > 1:
            with mp.get_context('fork').Pool(processes=nworkers) as pool:
                # local function to compute the solution at a given frequency, take only ind as argument
                fun = partial(self.compute_grad_freq, compute_points)
                res_grad = np.stack(pool.map(fun, freq_ind), axis=1)
        else:
            print("nworkers arg should be positive")
            exit(1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1))

        return res_grad
    
    def compute_corner_grad_freq(self, ind, adj=False):  # compute derivative at each corner, assumes each angle < pi
        """Return the gradient of the solution at each corner, taking the limit for particular solutions"""
        k = self.wave_numbers[ind]
        ampl, ampl_src, src_pos = self._select_sources(ind, adj)
        nsrc = len(self.vbound)

        diffs = self.poly.vertices[:, None, :] - self.vbound.bound[None, :, :]  # (nedges, npv, 2)
        norms = np.linalg.norm(diffs, axis=-1)  # (nedges, npv,)

        nder = np.sum(fundamental_solution_derivative(norms, k)[:, :, None] * diffs /
                      norms[:, :, None] * ampl[None, :nsrc, None], axis=1)

        diffs = self.poly.vertices[:, None, :] - src_pos[None, :, :]  # (nedges, nsrc, 2)
        norms = np.linalg.norm(diffs, axis=-1)  # (nedges, nsrc,)
        nder += np.sum(fundamental_solution_derivative(norms, k)[:, :, None] * diffs /
                       norms[:, :, None] * ampl_src[None, :, None], axis=1)

        recentered_int = np.arange(self.poly.nedges)
        trunc_recentered = np.zeros(self.poly.nedges - 1, dtype=int)
        if self.add_src > 0:
            for i in range(self.poly.nedges):
                # extract indices of the particular solutions that are not at the corner i
                trunc_recentered[:i], trunc_recentered[i:] = recentered_int[:i], recentered_int[i+1:]
                angles, diffs, norms = self.poly.compute_angles(self.poly.vertices[i][None, :])
                factor = self.factor[:, trunc_recentered]
                for t in range(self.add_src):

                    nder[i] += np.sum(particular_solution_gradient(norms[:, trunc_recentered],
                                                                   angles[:, trunc_recentered], k, t * factor,
                                                                   self.poly.polar_angles[trunc_recentered])[0] *
                                      ampl[trunc_recentered + (nsrc + t*self.poly.nedges), None], axis=0)
        return nder

    def compute_corner_grad(self, freq_ind=None, adj=False, nworkers=1, verbose=False):
        t1 = time.time()
        if freq_ind is None:  # compute all frequencies if not specified
            freq_ind = np.arange(1, self.nfreqs)

        if nworkers == 1:
            res_grad = np.zeros([self.poly.nedges, len(freq_ind), 2], dtype=np.complex128)

            for k, i in enumerate(freq_ind):
                res_grad[:, k] = self.compute_corner_grad_freq(i, adj=adj)
        elif nworkers > 1:
            with mp.get_context('fork').Pool(processes=nworkers) as pool:
                # local function to compute the solution at a given frequency, take only ind as argument
                fun = partial(self.compute_corner_grad_freq, adj=adj)
                res_grad = np.stack(pool.map(fun, freq_ind), axis=1)
        else:
            print("nworkers arg should be positive")
            exit(1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1))

        return res_grad

    def compute_corner_values(self, freq_ind=None, adj=False):
        return self.compute(self.poly.vertices, freq_ind, adj=adj), self.compute_corner_grad(freq_ind, adj=adj)

    def boundary_operator(self, ind, bound, aggregate=False, adjoint=False):
        """Compute the Robin operator dp/dn + ik p on the given boundary at the given frequency indices for the 
        approximated solution, WITHOUT the source contribution (which is computed using boundary_operator_source)."""
        k = self.wave_numbers[ind]
        if adjoint:
            beta, ampl = self.beta_adj, self.ampl_adj[ind]
        else:
            beta, ampl = self.beta, self.ampl[ind]
        res = np.zeros((len(bound), self.get_nvirt()), dtype=complex)  # (nbound, nvbound + nedges*add_src)
        res[:, :len(self.vbound)] = robin_operator(bound.bound, bound.normvec, ampl[:len(self.vbound)],
                                                   self.vbound.bound, k, beta)  # virtual sources contributions,

        angles, _, norms = self.poly.compute_angles(bound.bound)
        indstart, indstop = len(self.vbound),  len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):  # particular solutions contributions
            grad = particular_solution_gradient(norms, angles, k, t * self.factor,
                                                self.poly.polar_angles)  # (np, nvertices, 2)
            res[:, indstart:indstop] = ((np.sum(grad * bound.normvec[:, None, :], axis=-1) + beta *  # (np, nvertices, )
                                         k * 1j * particular_solution(norms, angles, k, t * self.factor)) *
                                        ampl[indstart:indstop].reshape(1, -1))
            indstart, indstop = indstop, indstop + self.poly.nedges

        if aggregate:
            res = np.sum(res, axis=1)
        return res

    def boundary_operator_source(self, ind, bound, adjoint=False):
        """Robin operator dp/dn + ik p applied at the source."""
        if adjoint:
            beta, src_pos, ampl = self.beta_adj, self.mic_pos, self.src_ampl_adj
        else:
            beta, src_pos, ampl = self.beta, self.src_pos, self.src_ampl

        return np.squeeze(robin_operator(bound.bound, bound.normvec, ampl, src_pos.reshape((-1, 2)),
                                         self.wave_numbers[ind], beta))

    def plot_bound_values(self, ind, plot3d=True, show=True, adjoint=False):
        plot_nodes = self.error_bound.bound
        mfs_val = self.boundary_operator(ind, self.error_bound, adjoint=adjoint,
                                         aggregate=True)
        src_val = - self.boundary_operator_source(ind, self.error_bound, adjoint=adjoint)

        error_real = np.sqrt(np.mean(np.abs(np.real(mfs_val) - np.real(src_val)) ** 2))
        error_imag = np.sqrt(np.mean(np.abs(np.imag(mfs_val) - np.imag(src_val)) ** 2))

        if plot3d:
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_title("Real part, nsrc={}, nb={}, error={:3f}".format(self.get_nvirt(), self.get_nbound(), error_real))
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

            max_range = max(plot_nodes[:, 0].max()-plot_nodes[:, 0].min(),
                            plot_nodes[:, 1].max()-plot_nodes[:, 1].min())
            mid = np.mean(self.poly.vertices, axis=0)
            for ax in [ax1, ax2]:
                ax.set_xlim([mid[0] - max_range/2, mid[0] + max_range/2])
                ax.set_ylim([mid[1] - max_range/2, mid[1] + max_range/2])

            plt.legend()

        else:
            fig, ax = plt.subplots(2, 2, figsize=(20, 6))
            ax[0, 0].set_ylim([np.min(src_val.real), np.max(src_val.real)])
            ax[0, 1].set_ylim([np.min(src_val.imag), np.max(src_val.imag)])

            ax[0, 0].set_title("Real part, nsrc={}, mean error={:3f}".format(self.get_nvirt(), error_real))
            ax[0, 1].set_title("Imaginary part, mean error={:3f}".format(error_imag))
            ax[0, 0].plot(np.real(mfs_val), label="MFS")
            ax[0, 0].plot(np.real(src_val), '--', label="Source contribution")
            ax[0, 1].plot(np.imag(mfs_val), label="MFS")
            ax[0, 1].plot(np.imag(src_val), '--', label="Source contribution")

            # add horizontal line at 0
            ax[0, 0].plot([0, len(mfs_val)], [0, 0], 'k')
            ax[0, 1].plot([0, len(mfs_val)], [0, 0], 'k')
            # add vertical bars at each edge change
            plt.legend()
            plt.tight_layout()

            # add a subplot under both plots to show the coefficients
            ax3 = fig.add_subplot(212)
            ax3.set_title("Coefficients")
            ax3.semilogy(np.abs(self.ampl[ind].real), '.', label="real part")
            ax3.semilogy(np.abs(self.ampl[ind].imag), '.', label="imaginary part")
            plt.legend()
            if show:
                plt.show()
        plt.show()


if __name__ == '__main__':
    pass