import numpy as np
from mfs import (SampledBoundary, particular_solution_gradient, robin_operator, particular_solution,
                 fundamental_solution, fundamental_solution_derivative, robin_operator_efficient)
from utils import Polygon, c
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class Simulation:
    def __init__(self, vertices: np.ndarray, fs: float, nfreq: int, dx: float, pos_src: np.ndarray, mic_pos: np.ndarray,
                 observations=None, add_src=15, nsrc_schedule=None, min_dist=0.01, error_thresh=0.05, sep_red=0.8,
                 src_ampl=None, beta: float | np.ndarray = 0.1, 
                 origin=None, vmin_dist=0, nworkers=1, simulate=False, max_tries=5):
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
        assert nworkers > 0, "nworkers should be positive"

        if src_ampl is None:
            src_ampl = np.ones((len(pos_src), nfreq), dtype=np.complex128)
        else:
            assert src_ampl.shape == (len(pos_src), nfreq), "src_ampl should have shape (nfreq, len(pos_src))"
        self.src_ampl = src_ampl
        self.src_pos = np.reshape(pos_src, [-1, 2])
        self.vmin_dist = vmin_dist

        if isinstance(beta, float):  # same beta for every wall
            beta = np.full(len(vertices), beta)

        self.beta: np.ndarray = beta
        self.beta_adj: np.ndarray = -beta  # beta coefficients for state and adjoint

        self.fs, self.dx = fs, dx   # maximal frequency and distance between real and virtual boundaries

        self.poly = Polygon(vertices, origin=origin)
        self.factor = np.pi / self.poly.angles[None, :]  # angular factor for particular solutions

        self.freqs = np.linspace(0, fs, nfreq)
        self.wave_numbers = 2 * np.pi * self.freqs / c
        self.nfreqs = nfreq

        self.min_dist = min_dist  # min distance from boundary points to the corners of the polygon on each edge
        self.add_src = add_src  # nnumber of additional sources in the enriched MFS basis at each corner

        # amplitudes for the source terms in the adjoint equation
        self.src_ampl_adj = np.zeros([len(mic_pos), self.nfreqs], dtype=complex)
        self.nworkers = nworkers

        if observations is not None:
            self.obs = observations
        else:  # create dummy observations
            self.obs = np.zeros([len(mic_pos), self.nfreqs], dtype=complex)

        self.mic_pos = mic_pos  # microphone positions

        # will be used to store precomputed values
        self._diffs, self._norms, self._diffs_src, self._norms_src = None, None, None, None
        self._angles_part, self._norms_part = None, None

        if nsrc_schedule is None:
            # sampling strategy : 3 points per wavelength
            nsrc = 10
            nsrc_schedule = dict(method='per_wavelength', nsrc=nsrc)

        if nsrc_schedule["method"] not in ["per_wavelength", "separation"]:
            raise ValueError("sampling method should be either per_wavelength or separation")

        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = c / self.freqs[-1] / (nsrc_schedule["nsrc"] - 1)
        self.vbound = SampledBoundary(self.poly, nsrc_schedule["sep"],
                                      min_dist=0., sample_type="separation", dx=self.dx)
        self.ampl = np.zeros([self.get_nvirt(), self.nfreqs], dtype=complex)

        # check if there are enough boundary points, take the max number of virtual sources if not
        nbound = self.get_nvirt()*1.1
        # sample boundary
        self.room_bound = SampledBoundary(self.poly, nbound, min_dist=min_dist, sample_type="total")
        print("Number of boundary points: ", self.get_nbound(), flush=True)

        nerror = 1.3 * nbound
        # nodes for computing error
        self.error_bound = SampledBoundary(self.poly, nerror, min_dist=min_dist, sample_type="total")
        print("Number of error points: ", len(self.error_bound), flush=True)

        print("simulating frequency ", self.freqs[-1], flush=True)
        rel_error = np.inf
        t1 = time.time()
        while rel_error > error_thresh and max_tries > 0:  # increase number of sources until precision threshold is met
            max_tries -= 1
            self.simulate(np.array([-1]), verbose=True, adjoint=False)

            # check error
            rel_error = self.evaluate_error(-1, adjoint=False)

            if rel_error > error_thresh:  # resample with more sources
                nsrc_schedule["sep"] = nsrc_schedule["sep"] * sep_red

                self.vbound = SampledBoundary(self.poly, nsrc_schedule["sep"],
                                              min_dist=self.vmin_dist, sample_type="separation", dx=self.dx)
                self.ampl = np.zeros([self.get_nvirt(), self.nfreqs], dtype=complex)
                print("error {} above threshold, retrying "
                      "with separation {}, {} sources".format(rel_error, nsrc_schedule["sep"],
                                                              len(self.vbound)), flush=True)

                totnsrc = len(self.vbound) + self.add_src * self.poly.nedges
                if totnsrc > nbound:
                    nbound = totnsrc*1.1
                    print("error above threshold, increasing number of boundary points", flush=True)
                    self.room_bound = SampledBoundary(self.poly, nbound, min_dist=min_dist, sample_type="total")

                    nerror = 1.3 * nbound
                    self.error_bound = SampledBoundary(self.poly, nerror, min_dist=min_dist, sample_type="total")

            if max_tries == 0 and rel_error > error_thresh:
                print("error threshold not reached, simulation might be inaccurate", flush=True)

            t2 = time.time()
            print("time {} s".format(t2-t1), flush=True)
            print("number of sources: {} relative error: {}".format(self.get_nvirt(), rel_error), flush=True)

        self.nsrc_schedule = nsrc_schedule
        # amplitudes for adjoint virtual sources
        self.ampl_adj = np.zeros([self.get_nvirt(), self.nfreqs], dtype=complex)
        if simulate and nfreq > 2:   # nfreq - 1 because the last freq is already computed
            self.simulate(ind=np.arange(1, nfreq-1), adjoint=False, verbose=True, reset_ampl=False)

    def get_nbound(self):
        return len(self.room_bound)

    def get_nvirt(self):
        return len(self.vbound) + self.add_src * self.poly.nedges

    def update_poly(self, new_vert, bounds=None, check_in=None, convex_proj=False, min_length=None, max_angle=None,
                    increase_factor=1.1):
        nvert_old = self.poly.nedges
        success = self.poly.update_vertices(new_vert, bounds=bounds, check_in=check_in, convex_proj=convex_proj,
                                            update_origin=True, min_length=min_length, max_angle=max_angle)

        if success:  # resample polygon after update
            self.vbound = SampledBoundary(self.poly, self.nsrc_schedule["sep"], min_dist=0., dx=self.dx)
            self.room_bound = SampledBoundary(self.poly, sample_param=self.get_nvirt()*increase_factor,
                                              dx=0, min_dist=self.min_dist, sample_type='total')
            self.error_bound = SampledBoundary(self.poly, sample_param=self.get_nbound()*1.3, dx=0,
                                               min_dist=self.min_dist, sample_type='total')
            self.factor = np.pi / self.poly.angles[None, :]  # angular factor for particular solutions
            self.ampl = np.zeros([self.get_nvirt(), self.nfreqs], dtype=complex)
            self.ampl_adj = np.zeros([self.get_nvirt(), self.nfreqs], dtype=complex)
            
            # if the number of vertices change: handle betas as if it was constant
            if self.poly.nedges != nvert_old:
                self.beta = np.full(self.poly.nedges, self.beta[0])
                self.beta_adj = np.full(self.poly.nedges, self.beta_adj[0])

        else:
            print("Failure to update the polygon", flush=True)
        return success

    def _select_sources(self, ind, adjoint):
        if adjoint:
            return self.ampl_adj[:, ind], self.src_ampl_adj[:, ind], self.mic_pos
        else:
            return self.ampl[:, ind], self.src_ampl[:, ind], self.src_pos

    def compute(self, points, ind=None, adjoint=False, verbose=False):
        """Compute the values taken by the state p or the adjoint q at each microphone position."""
        if ind is None:
            ind = np.arange(1, self.nfreqs)

        # precompute norms to reduce the number of calculations
        self._precompute_diffs(points, self.vbound.bound), self._precompute_diffs_src(points, adjoint)

        t1 = time.time()
        if self.nworkers == 1:
            res = np.full([len(points), len(ind)], np.nan, dtype=np.complex128)
            for k, i in enumerate(ind):
                res[:, k] = self._compute_freq(i, adjoint)
        else:
            res = np.stack(Parallel(n_jobs=self.nworkers)(delayed(self._compute_freq)(i, adjoint) for i in ind), axis=1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1), flush=True)

        return res

    def _compute_freq(self, ind, adjoint=False):
        ampl, ampl_src, src_pos = self._select_sources(ind, adjoint)

        # contribution from virtual sources
        res = np.sum(ampl[None, :len(self.vbound)] *
                     fundamental_solution(self._norms, self.wave_numbers[ind]), axis=1)
        # add contribution from particular solutions
        indstart, indstop = len(self.vbound), len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):  # particular solutions contributions
            res += np.sum(particular_solution(self._norms_part, self._angles_part,
                                              self.wave_numbers[ind], t * self.factor) *
                          ampl[indstart:indstop].reshape(1, -1), axis=-1)
            indstart, indstop = indstop, indstop + self.poly.nedges

        # add source contribution
        res += np.sum(ampl_src[None, :]*fundamental_solution(self._norms_src, self.wave_numbers[ind]), axis=-1)

        return res

    def _compute_grad_freq(self, ind, adjoint=False):
        """Compute the gradient of the solution at the given points (shape (npoints, d)) at the given frequency"""
        ampl, ampl_src, src_pos = self._select_sources(ind, adjoint)

        k = self.wave_numbers[ind]
        grad = np.sum(fundamental_solution_derivative(self._norms, k)[:, :, None] * self._diffs
                      / self._norms[:, :, None] * ampl[None, :len(self.vbound), None], axis=1)  # (np, 2)

        # add source contribution
        grad += np.sum(fundamental_solution_derivative(self._norms_src, k)[:, :, None] * self._diffs_src
                       / self._norms_src[:, :, None] * ampl_src[None, :, None], axis=1)
        indstart, indstop = len(self.vbound), len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):
            grad += np.sum(particular_solution_gradient(self._norms_part, self._angles_part, k, t * self.factor,
                                                        self.poly.polar_angles) * ampl[None, indstart:indstop, None],
                           axis=1)
            indstart, indstop = indstop, indstop + self.poly.nedges

        return grad  # shape (npoints, nfreq, 2

    def _simulate_freq(self, ind, adjoint=False, verbose=False):
        """Assemble and solve the MFS system for the frequency of index ind, return the vector of corresponding
        amplitudes (sources and particular solutions)."""
        t1 = time.time()

        # assemble the system matrix and rhs
        A = self._boundary_operator_efficient(ind, adjoint=adjoint)  # amplitudes have already been reset to 1 to compute this

        # source contribution
        b = - np.reshape(self._boundary_operator_efficient_src(ind, adjoint=adjoint), [-1, 1])
        
        try:
            res = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            print("solve error, freq:", 343 * self.wave_numbers[ind] / np.pi / 2, flush=True)
            exit(1)

        t2 = time.time()
        if verbose:
            print("simulating frequency {}; time {} s; {} sources".format(self.freqs[ind], t2-t1, self.get_nvirt()), flush=True)

        return res.squeeze()

    def evaluate_error(self, freq_ind, adjoint=False):
        """Checks if the solution satisfies the boundary condition at the error nodes."""
        mfs_val = self.boundary_operator(freq_ind, self.error_bound, aggregate=True, adjoint=adjoint)
        src_val = -self.boundary_operator_source(freq_ind, self.error_bound, adjoint=adjoint)

        rel_error = np.sqrt(np.sum((np.abs(mfs_val - src_val)) ** 2)) / np.linalg.norm(src_val)
        return rel_error

    def get_ind(self, fmin, fmax):
        """Return indices from the range of frequencies [fmin, fmax]"""
        return np.where((self.freqs > fmin) & (self.freqs <= fmax))[0]

    def simulate(self, ind=None, adjoint=False, verbose=True, reset_ampl=False):
        """Compute the solution coefficients at each frequency using nworkers processes."""
        if ind is None:  # compute all frequencies if not specified
            ind = np.arange(1, self.nfreqs)

        # precompute norms to reduce the number of calculations
        self._precompute_diffs(self.room_bound.bound, self.vbound.bound)
        self._precompute_diffs_src(self.room_bound.bound, adjoint)

        if adjoint:
            ampl = self.ampl_adj
        else:
            ampl = self.ampl

        # reset amplitudes to 1 to compute virtual sources contribution
        ampl[:, ind] = 1

        t1 = time.time()
        if self.nworkers == 1:  # 1 frequency at a time
            for i in ind:
                ampl[:, i] = self._simulate_freq(i, verbose=verbose, adjoint=adjoint)

        else:
            ampl_pool = Parallel(n_jobs=self.nworkers)(delayed(self._simulate_freq)(i, adjoint, verbose) for i in ind)

            for i, ii in enumerate(ind):
                ampl[:, ii] = ampl_pool[i]

        if not adjoint:  # update the adjoint amplitudes
            if reset_ampl:
                self.src_ampl_adj[:] = 0  # reset adjoint amplitudes
            self.src_ampl_adj[:, ind] = self.obs[:, ind] - self.compute(self.mic_pos, ind, adjoint=False)
        if verbose:
            print("total simulation time : {} s".format(time.time()-t1), flush=True)

    def compute_grad(self, points, ind=None, adjoint=False, verbose=False):
        """Compute the gradient of the solution at the given points (shape (npoints, d)) at the given frequency.
        Warning : should not be called at the vertices positions if the number of particular solutions is > 0."""
        _, _, src_pos = self._select_sources(0, adjoint)

        # precompute norms to reduce the number of calculations
        self._precompute_diffs(points, self.vbound.bound), self._precompute_diffs_src(points, adjoint)

        t1 = time.time()
        if ind is None:  # compute all frequencies if not specified
            ind = np.arange(1, self.nfreqs)

        if self.nworkers == 1:
            res_grad = np.zeros([len(points), len(ind), 2], dtype=np.complex128)

            for k, i in enumerate(ind):
                res_grad[:, k] = self._compute_grad_freq(i, adjoint=adjoint)
        else:
            res_grad = Parallel(n_jobs=self.nworkers)(delayed(self._compute_grad_freq)(i, adjoint) for i in ind)
            res_grad = np.stack(res_grad, axis=1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1), flush=True)

        return res_grad

    def _compute_corner_grad_freq(self, ind, adjoint=False):
        """Return the gradient of the solution at each corner"""
        k = self.wave_numbers[ind]
        ampl, ampl_src, src_pos = self._select_sources(ind, adjoint)
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

    def compute_corner_grad(self, ind=None, adjoint=False, verbose=False):
        t1 = time.time()
        if ind is None:  # compute all frequencies if not specified
            ind = np.arange(1, self.nfreqs)

        if self.nworkers == 1:
            res_grad = np.zeros([self.poly.nedges, len(ind), 2], dtype=np.complex128)

            for k, i in enumerate(ind):
                res_grad[:, k] = self._compute_corner_grad_freq(i, adjoint=adjoint)
        else:
            res_grad = Parallel(n_jobs=self.nworkers)(delayed(self._compute_corner_grad_freq)(i, adjoint) for i in ind)
            res_grad = np.stack(res_grad, axis=1)

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1), flush=True)

        return res_grad

    def compute_corner_values(self, freq_ind=None, adjoint=False):
        return (self.compute(self.poly.vertices, freq_ind, adjoint=adjoint),
                self.compute_corner_grad(freq_ind, adjoint=adjoint))

    def boundary_operator(self, ind, bound, aggregate=False, adjoint=False):
        """Compute the Robin operator dp/dn + ik p on the given boundary at the given frequency indices for the
        approximated solution, WITHOUT the source contribution (which is computed using boundary_operator_source)."""
        k = self.wave_numbers[ind]
        if adjoint:
            beta, ampl = self.beta_adj, self.ampl_adj[:, ind]
        else:
            beta, ampl = self.beta, self.ampl[:, ind]

        beta_loc = beta[bound.bound_ids, None]
        res = np.zeros((len(bound), self.get_nvirt()), dtype=complex)  # (nbound, nvbound + nedges*add_src)
        res[:, :len(self.vbound)] = robin_operator(bound.bound, bound.normvec, ampl[:len(self.vbound)],
                                                   self.vbound.bound, k, beta_loc)  # virtual sources contributions,

        angles, _, norms = self.poly.compute_angles(bound.bound)
        indstart, indstop = len(self.vbound),  len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):  # particular solutions contributions
            grad = particular_solution_gradient(norms, angles, k, t * self.factor,
                                                self.poly.polar_angles)  # (np, nvertices, 2)
            res[:, indstart:indstop] = ((np.sum(grad * bound.normvec[:, None, :], axis=-1) + beta_loc *  # (np, nvertices, )
                                         k * 1j * particular_solution(norms, angles, k, t * self.factor)) *
                                        ampl[indstart:indstop].reshape(1, -1))
            indstart, indstop = indstop, indstop + self.poly.nedges

        if aggregate:
            res = np.sum(res, axis=1)
        return res

    def _boundary_operator_efficient(self, ind, adjoint=False):
        """Compute the Robin operator at room_bound, using precomputed values for the differences and norms."""
        k = self.wave_numbers[ind]
        if adjoint:
            beta, ampl = self.beta_adj, self.ampl_adj[:, ind]
        else:
            beta, ampl = self.beta, self.ampl[:, ind]
        beta_loc = beta[self.room_bound.bound_ids, None]
        res = np.zeros((self.get_nbound(), self.get_nvirt()), dtype=complex)  # (nbound, nvbound + nedges*add_src)
        res[:, :len(self.vbound)] = robin_operator_efficient(self.room_bound.normvec, self._diffs, self._norms,
                                                             ampl[:len(self.vbound)], k, beta_loc)

        indstart, indstop = len(self.vbound), len(self.vbound) + self.poly.nedges
        for t in range(self.add_src):  # particular solutions contributions
            grad = particular_solution_gradient(self._norms_part, self._angles_part, k, t * self.factor,
                                                self.poly.polar_angles)  # (np, nvertices, 2)
            res[:, indstart:indstop] = ((np.sum(grad * self.room_bound.normvec[:, None, :], axis=-1) +  # np, nvertices
                                         beta_loc * k * 1j * particular_solution(self._norms_part, self._angles_part,
                                                                             k, t * self.factor)) *
                                        ampl[indstart:indstop].reshape(1, -1))
            indstart, indstop = indstop, indstop + self.poly.nedges

        return res

    def boundary_operator_source(self, ind, bound, adjoint=False):
        """Robin operator dp/dn + ik p applied at the source."""
        if adjoint:
            beta, src_pos, ampl = self.beta_adj, self.mic_pos, self.src_ampl_adj[:, ind]
        else:
            beta, src_pos, ampl = self.beta, self.src_pos, self.src_ampl[:, ind]
        beta_loc = beta[bound.bound_ids, None]
        
        return np.squeeze(np.sum(robin_operator(bound.bound, bound.normvec, ampl, src_pos.reshape((-1, 2)),
                                                self.wave_numbers[ind], beta_loc), axis=1))

    def _boundary_operator_efficient_src(self, ind, adjoint=False):
        """Compute the Robin operator at room_bound for the source term, using precomputed values for the differences
        and norms."""
        k = self.wave_numbers[ind]
        if adjoint:
            beta, ampl = self.beta_adj, self.src_ampl_adj[:, ind]
        else:
            beta, ampl = self.beta, self.src_ampl[:, ind]
        beta_loc = beta[self.room_bound.bound_ids, None]

        return np.squeeze(np.sum(robin_operator_efficient(self.room_bound.normvec, self._diffs_src, self._norms_src,
                                                          ampl, k, beta_loc), axis=1))

    def _precompute_diffs(self, p1, p2):
        """Precompute the differences, norms and angles between p1 and p2 (p2 refers to sources). The angles are
        computed relatively to p1."""
        self._diffs = p1[:, None, :] - p2[None, :, :]  # (np, nsrc, 2)
        self._norms = np.linalg.norm(self._diffs, axis=-1)  # (np, nsrc)
        self._angles_part, _, self._norms_part = self.poly.compute_angles(p1)

    def _precompute_diffs_src(self, points, adjoint):
        """Precompute the differences, norms and angles between points and the sources."""
        if adjoint:
            pos_src = self.mic_pos
        else:
            pos_src = self.src_pos
        self._diffs_src = points[:, None, :] - pos_src[None, :, :]
        self._norms_src = np.linalg.norm(self._diffs_src, axis=-1)  # (np, nsrc)

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
            ax3.semilogy(np.abs(self.ampl[:, ind].real), '.', label="real part")
            ax3.semilogy(np.abs(self.ampl[:, ind].imag), '.', label="imaginary part")
            plt.legend()
            if show:
                plt.show()
        plt.show()


if __name__ == '__main__':
    # create a triangle with an angle alpha at the origin
    alpha, alpha_rot = np.pi*0.95,  np.pi*0.5
    vert = np.array([[0, 0.], [1., 0.], [0,1.],[np.cos(alpha), np.sin(alpha)]])
    # apply rotation
    rot = np.array([[np.cos(alpha_rot), -np.sin(alpha_rot)],
                    [np.sin(alpha_rot), np.cos(alpha_rot)]])
    vert = np.dot(vert, rot)
    p = Polygon(vert)
    src_angle = alpha * np.sqrt(2)/4
    pos_src = np.array([[np.cos(src_angle), np.sin(src_angle)]]) * np.sqrt(3)/4.
    mic_pos = np.array([[np.cos(np.pi/np.sqrt(5)), np.sin(np.pi/np.sqrt(5))]])/2
    pos_src, mic_pos = np.dot(pos_src, rot), np.dot(mic_pos, rot)
    plt.plot(p.vertices[:, 0], p.vertices[:, 1], 'o-')
    plt.plot(pos_src[0, 0], pos_src[0,1], 'o', color='red')
    plt.plot(mic_pos[0, 0], mic_pos[0, 1], 'o', color='green')
    plt.axis('equal')
    # plt.plot(points[:, 0], points[:, 1], 'x')

    plt.show()
    pos_src = np.mean(p.vertices, axis=0).reshape((1, 2))

    sim = Simulation(p.vertices, mic_pos=mic_pos, min_dist=0.005, dx=0.3, fs=256, nfreq=4,
                          pos_src=pos_src, nsrc_schedule=dict(method='per_wavelength', nsrc=30),
                          error_thresh=1e-5, beta=.1, add_src=10, observations=np.ones([1, 4]))
    sim.simulate()
    print(sim.poly.angles)
    print(sim.compute_grad(sim.mic_pos))