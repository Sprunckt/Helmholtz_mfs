import numpy as np
from mfs import HelmholtzMFS, SampledBoundary, compute_hessian2d
from utils import Polygon, c, ExtrudedPolygon, in_polygon, Cuboid
import multiprocessing as mp
import time
from scipy import signal
from functools import partial
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, vertices: np.ndarray, fs: float, nfreq: int, boundsep: float, dx: float, pos_src: np.ndarray,
                 cond='neumann', add_src=15, nsrc_schedule=None, min_dist=0.01, error_sep=None, error_thresh=0.05,
                 sep_red=0.8, src_ampl=None, beta=None, height=0., poly_virt=None, convexify=False, origin=None):
        """
        Simulation class for Helmholtz equation: simulate a source at pos_src in a polygonal domain with boundary
        conditions at different frequencies. Can generate time-domain signals by applying an inverse Fourier transform.
        Args:
            - vertices: vertices of the polygonal domain
            - fs: sampling frequency
            - nfreq: number of frequencies
            - boundsep: distance between boundary points of the room walls
            - dx: distance between real and virtual boundaries
            - pos_src: position of the source
            - cond: boundary condition (neumann, dirichlet, robin)
            - add_src: number of additional sources in the enriched MFS basis
            - nsrc_schedule: strategy to sample the virtual boundary. Should contain a key "sampling_method" which can
        be either "separation" or "per_wavelength". The key "freq_marks" should refer to an array of frequencies at
        which to resample the virtual boundary. If the method is "per_wavelength", the key "nsrc" should contain an
        array of number of sources per wavelength at each frequency mark used to compute separation. If the method is
        "separation", the separations should be given using the key "sep".
            - min_dist: minimum distance between boundary points and the corners of the polygon
            - error_sep: separation between sources used to verify the boundary error at each freq_mark is under the
        threshold
            - error_thresh: threshold for the boundary error at initialization
            - sep_red: reduction factor applied at the separation between sources at each resampling in case the error
        is above the threshold
        """
        if src_ampl is None:
            src_ampl = np.ones([len(pos_src), nfreq], dtype=np.complex128)
        else:
            assert src_ampl.shape == (len(pos_src), nfreq), "src_ampl should have shape (len(pos_src), nfreq)"
        src_ampl[:, 0] = 0.  # zero amplitude for freq 0
        self.src_ampl = src_ampl

        self.fs, self.dx = fs, dx
        self.convexify = convexify
        if height > 0 or np.asarray(vertices).shape == (3,):
            self.d = 3
            if height > 0:
                self.poly = ExtrudedPolygon(vertices, height)
                if poly_virt is not None:
                    self.poly_virt = poly_virt
                elif convexify:
                    print("Convex projection for 3d polyhedras unsupported")
                    exit(-1)
                else:
                    self.poly_virt = self.poly
            else:  # cuboid
                self.poly = Cuboid(*vertices)
                self.poly_virt = self.poly
        else:
            self.d = 2
            self.poly = Polygon(vertices, origin=origin)
            if poly_virt is not None:
                self.poly_virt = poly_virt
            elif convexify:
                self.poly_virt.convex_projection()
            else:
                self.poly_virt = self.poly

        self.freqs = np.linspace(0, fs, nfreq)
        self.wave_numbers = 2 * np.pi * self.freqs / c
        self.nfreqs = nfreq
        self.min_dist = min_dist
        # ampl will contain the amplitudes for each source at each freq, vpos the boundary positions at each mark
        self.ampl, self.vpos = [None for _ in range(self.nfreqs)], []

        self._to_simulate = np.ones(nfreq, dtype=bool)
        self._to_simulate[0] = False

        if nsrc_schedule is None:
            # sampling strategy : 3 points per wavelength for f > 100, otherwise 10. Resample every 200 Hz or so
            freq_marks = np.array([100.])
            nsrc = np.array([10, 3.])
            nsrc[0], nsrc[1:] = 10, 3
            nsrc_schedule = dict(freq_marks=freq_marks, method='per_wavelength', nsrc=nsrc)

        # generate sample_marks from freq_marks, ie indices of frequencies where to sample
        if nsrc_schedule.get("sample_marks") is None:
            nsrc_schedule["freq_marks"][-1] = np.clip(nsrc_schedule["freq_marks"][-1], 0, self.freqs[-1])

            nsrc_schedule["sample_marks"] = np.argmax(self.freqs[np.newaxis, :] >=
                                                      nsrc_schedule["freq_marks"][:, np.newaxis], axis=1)

        if nsrc_schedule["method"] not in ["per_wavelength", "separation"]:
            raise ValueError("sampling method should be either per_wavelength or separation")

        # add last sample mark (highest frequency)
        if len(nsrc_schedule["sample_marks"]) < len(nsrc_schedule["freq_marks"]) + 1:
            nsrc_schedule["sample_marks"] = np.concatenate([nsrc_schedule["sample_marks"], [len(self.freqs) - 1]])
        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = np.zeros(len(nsrc_schedule["sample_marks"]))

        for i in range(len(nsrc_schedule["sample_marks"])):  # sample virtual boundary for each mark
            if nsrc_schedule["method"] == "per_wavelength":
                nsrc_schedule["sep"][i] = c / self.freqs[nsrc_schedule["sample_marks"][i]] / \
                                           (nsrc_schedule["nsrc"][i] - 1)
            vbound = SampledBoundary(self.poly_virt, nsrc_schedule["sep"][i],
                                     min_dist=0., sample_type="separation", dx=self.dx, distribution='uniform')

            self.vpos.append(vbound)

        # for each frequency, assign the index of the corresponding reference frequency mark
        self.freq_ref = np.zeros(nfreq, dtype=int)
        self.freq_ref[:nsrc_schedule["sample_marks"][0] + 1] = nsrc_schedule["sample_marks"][0]
        for i in range(1, len(nsrc_schedule["sample_marks"])):
            self.freq_ref[nsrc_schedule["sample_marks"][i-1] + 1:
                          nsrc_schedule["sample_marks"][i] + 1] = nsrc_schedule["sample_marks"][i]

        # convert to integers between 0 and nmarks-1
        _, self.freq_ref = np.unique(self.freq_ref, return_inverse=True)

        # check if there are enough boundary points, take the max number of virtual sources if not
        if boundsep > nsrc_schedule["sep"][-1]:
            boundsep = nsrc_schedule["sep"][-1]*0.95
            print("insufficient boundary separation to account for all frequencies, clipping")

        # sample boundary using boundsep
        room_bound = SampledBoundary(self.poly, boundsep, min_dist=min_dist, sample_type="separation")

        self.mfs = HelmholtzMFS(room_bound=room_bound, virtual_bound=self.vpos[0], pos_src=pos_src,
                                poly=self.poly, cond=cond, add_fun=add_src, beta=beta,
                                src_ampl=self.src_ampl[:, nsrc_schedule["sample_marks"][0]])

        print("Number of boundary points: ", self.mfs.nbound)

        if error_sep is None:
            error_sep = boundsep * 0.8
        elif error_sep > boundsep:
            error_sep = boundsep * 0.8
            print("error_sep is larger than boundsep, clipping")

        # nodes for computing error
        error_points, _, error_ids = self.poly.sample(error_sep, sample_type="separation", min_dist=min_dist)
        print("Number of error points: ", len(error_points))
        for i, indfreq in enumerate(nsrc_schedule["sample_marks"]):
            print("simulating frequency ", self.freqs[indfreq])
            k = 2 * np.pi * self.freqs[indfreq] / c  # update wave number
            self.mfs.src_ampl = self.src_ampl[:, indfreq]
            # simulate at each mark frequency, increase number of sources until error is below threshold
            rel_error, max_tries = np.inf, 10

            if i > 0 and nsrc_schedule["sep"][i] > nsrc_schedule["sep"][i-1]:  # separation should decrease
                nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i-1]

                self.vpos[i] = self.vpos[i-1].copy()
            t1 = time.time()
            while rel_error > error_thresh and max_tries > 0:
                max_tries -= 1
                self.mfs.update_sources(self.vpos[i])  # update source positions
                self.mfs.mfs_fit(k=k, solver="lstsq")

                # check error
                mfs_val = self.mfs.boundary_operator(error_points, error_ids, k)
                src_val = -self.mfs.boundary_operator(error_points, error_ids, k, use_src=True)
                rel_error = np.sqrt(np.sum((np.abs(mfs_val - src_val)) ** 2)) / np.linalg.norm(src_val)
                if rel_error > error_thresh:  # resample with more sources
                    nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i] * sep_red

                    self.vpos[i] = SampledBoundary(self.poly_virt, nsrc_schedule["sep"][i],
                                                   min_dist=0., sample_type="separation", dx=self.dx,
                                                   distribution='uniform')

                    print("error {} above threshold, retrying "
                          "with separation {}, {} sources".format(rel_error, nsrc_schedule["sep"][i],
                                                                  len(self.vpos[i])))

                    if nsrc_schedule["sep"][i] < boundsep:
                        boundsep = nsrc_schedule["sep"][i]*0.95
                        print("error above threshold, clipping boundary separation to minimal source separation")
                        room_bound = SampledBoundary(self.poly, boundsep, min_dist=min_dist, sample_type="separation")
                        self.mfs.update_boundary(room_bound)

                        if error_sep > boundsep:
                            error_sep = boundsep * 0.95
                            error_points, _, error_ids = self.poly.sample(error_sep, sample_type="separation", min_dist=min_dist)

                            print("error_sep is larger than boundsep, clipping ; {} points".format(len(error_points)))

            if max_tries == 0:
                print("error threshold not reached, simulation might be inaccurate")
            self.ampl[indfreq] = self.mfs.ampl.copy()
            self._to_simulate[indfreq] = False
            t2 = time.time()

            print("time {} s".format(t2-t1))
            print("number of sources: {} relative error: {}".format(self.mfs.nsrc, rel_error))

        self.boundsep = boundsep
        self.nsrc_schedule = nsrc_schedule
        self.ampl[0] = np.zeros(len(self.vpos[0].bound)+add_src*self.poly.nedges)  # to compute null frequency as 0

    def reset(self):
        self._to_simulate[1:] = True

    def _resample(self):
        for i in range(len(self.vpos)):
            self.vpos[i] = SampledBoundary(self.poly_virt, self.nsrc_schedule["sep"][i], min_dist=0., dx=self.dx)
        self.mfs.update_boundary(SampledBoundary(self.poly, self.boundsep, 
                                                 min_dist=self.min_dist, dx=0))

        nsrc_max = len(self.vpos[-1])
        while nsrc_max > self.mfs.nbound:
            self.boundsep *= 0.95
            self.mfs.update_boundary(SampledBoundary(self.poly, self.boundsep, 
                                                     min_dist=self.min_dist, dx=0))

    def load(self, room_bound, vpos, ampl):
        self.mfs.update_boundary(room_bound)
        self.vpos = vpos
        self.ampl = ampl

    def update_poly(self, new_vert, new_virt_vert=None, parametrization="vertices",
                    bounds=None, check_in=None, convex_proj=False, min_length=None):
        if parametrization == "vertices":
            success = self.poly.update_vertices(new_vert, bounds=bounds, check_in=check_in, convex_proj=convex_proj,
                                                update_origin=True, min_length=min_length)
            if new_virt_vert is not None:
                self.poly_virt.update_vertices(new_virt_vert, bounds=bounds, update_origin=True)
            elif self.convexify:
                if self.d == 2:
                    self.poly_virt.convex_projection()   # todo: ensure poly_virt is always a copy of self.poly for convexify
                else:
                    print("Convex projection for 3d polyhedras unsupported")
                    exit(-1)
            else:
                self.poly_virt = self.poly  
        elif parametrization == "half-planes":
            success = self.poly.update_param(new_vert, bounds=bounds, check_in=check_in, min_length=min_length)
            if new_virt_vert is not None:
                success = self.poly_virt.update_param(new_virt_vert, bounds=bounds)
            elif self.convexify:
                if self.d == 2:
                    self.poly_virt = Polygon(self.poly.vertices)
                    self.poly_virt.convex_projection()
                else:
                    print("Convex projection for 3d polyhedras unsupported")
                    exit(-1)
        if success:
            self.mfs.poly = self.poly
            self._resample()
        else:
            print("Failure to update the polygon")
        return success

    def update_source_term(self, new_pos_src):
        self.mfs.update_source_term(new_pos_src)

    def simulate_freq(self, ind, verbose=False):
        t1 = time.time()

        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly, src_ampl=self.src_ampl[:, ind],
                               cond=self.mfs.cond, beta=self.mfs.beta, add_fun=self.mfs.add_fun)
        mfs_tmp.mfs_fit(k=self.wave_numbers[ind], solver="lstsq")
        t2 = time.time()
        if verbose:
            print("simulating frequency {}; time {} s; {} sources".format(self.freqs[ind], t2-t1, mfs_tmp.nsrc))
        return mfs_tmp.ampl

    def compute_freq(self, compute_points, ind, verbose=False, use_src=True):
        """Compute the solution at the given points (shape (npoints, d)) at the given frequency"""
        t1 = time.time()
        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                               beta=self.mfs.beta, ampl=self.ampl[ind],
                               add_fun=self.mfs.add_fun, src_ampl=self.src_ampl[:, ind])
        if verbose:
            print("computing frequency {}; {} sources".format(self.freqs[ind], mfs_tmp.nsrc))

        res = mfs_tmp.evaluate_sol(compute_points, self.wave_numbers[ind], use_src=use_src)

        t2 = time.time()
        if verbose:
            print("time {} s".format(t2 - t1))

        return res

    def compute_grad_freq(self, compute_points, ind, verbose=False, use_src=True):
        """Compute the solution at the given points (shape (npoints, d)) at the given frequency"""
        t1 = time.time()
        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                               beta=self.mfs.beta, ampl=self.ampl[ind],
                               add_fun=self.mfs.add_fun, src_ampl=self.src_ampl[:, ind])
        if verbose:
            print("computing frequency {}; {} sources".format(self.freqs[ind], mfs_tmp.nsrc))
        res = mfs_tmp.evaluate_grad(compute_points, self.wave_numbers[ind], use_src=use_src)
        t2 = time.time()
        if verbose:
            print("time {} s".format(t2 - t1))

        return res

    def evaluate_error(self, freq_ind, nb_error_points=None):
        if nb_error_points is None:
            nb_error_points = len(self.mfs.room_bound)*1.1
        k = self.wave_numbers[freq_ind]
        error_points, _, error_ids = self.poly.sample(nb_error_points, min_dist=0.005)
        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[freq_ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                               beta=self.mfs.beta, ampl=self.ampl[freq_ind],
                               add_fun=self.mfs.add_fun, src_ampl=self.src_ampl[:, freq_ind])

        mfs_val = mfs_tmp.boundary_operator(error_points, error_ids, k)
        src_val = - mfs_tmp.boundary_operator(error_points, error_ids, k, use_src=True)
        rel_error = np.sqrt(np.sum((np.abs(mfs_val - src_val)) ** 2)) / np.linalg.norm(src_val)
        return rel_error

    def get_ind(self, fmin, fmax):
        """Return indices from the range of frequencies [fmin, fmax]"""
        return np.where((self.freqs > fmin) & (self.freqs <= fmax))[0]

    def set_freq_ind_to_simulate(self, ind_sim):
        self._to_simulate[:] = False
        self._to_simulate[ind_sim] = True

    def set_freq_to_simulate(self, fmin, fmax):
        """Set ONLY the frequencies in the range [fmin, fmax] to be computed"""
        self._to_simulate[:] = False
        ind = self.get_ind(fmin, fmax)
        self._to_simulate[ind] = True
        self._to_simulate[0] = False  # null frequency is never computed

    def simulate(self, nworkers=1, verbose=True):
        """Compute the solution coefficients at each frequency using nworkers processes."""
        t1 = time.time()
        ind_comp = np.where(self._to_simulate)[0]
        if nworkers == 1:  # 1 frequency at a time
            for ind in ind_comp:
                self.ampl[ind] = self.simulate_freq(ind, verbose=verbose)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            partfunc = partial(self.simulate_freq, verbose=verbose)
            ampl = pool.map(partfunc, ind_comp)
            pool.close()
            for i, ind in enumerate(ind_comp):
                self.ampl[ind] = ampl[i]
        if verbose:
            print("total simulation time : {} s".format(time.time()-t1))

    def compute(self, compute_points, nworkers=1, use_src=True, verbose=False, freq_ind=None):
        """Compute the solution at the given points (shape (npoints, d)) at all frequencies, returns an array
        res of shape (npoints, len(freq_ind)) containing the scalar solution at each compute
        point and computed frequency specified by freq_ind."""

        t1 = time.time()

        if freq_ind is None:  # compute all frequencies if not specified
            freq_ind = np.arange(self.nfreqs)

        if nworkers == 1:
            res = np.full([len(compute_points), len(freq_ind)], np.nan, dtype=np.complex128)
            for k, i in enumerate(freq_ind):
                res[:, k] = self.compute_freq(compute_points, i, use_src=use_src)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            # local function to compute the solution at a given frequency, take only ind as argument
            fun = partial(self.compute_freq, compute_points, use_src=use_src)
            res = np.stack(pool.map(fun, freq_ind), axis=1)
            pool.close()

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1))

        return res

    def compute_grad(self, compute_points, nworkers=1, use_src=True, verbose=False, freq_ind=None):
        """Compute the solution at the given points (shape (npoints, d)) at all frequencies, returns a tuple
        (res, is_in) of arrays of shape (npoints, nfreqs) and (npoints,) respectively, where is_in is True if the
        corresponding point is inside the domain, False otherwise and res contains the scalar solution at each compute
        point and frequency."""

        t1 = time.time()
        if freq_ind is None:  # compute all frequencies if not specified
            freq_ind = np.arange(self.nfreqs)

        if nworkers == 1:
            res_grad = np.zeros([len(compute_points), len(freq_ind), 2], dtype=np.complex128)

            for k, i in enumerate(freq_ind):
                res_grad[:, k] = self.compute_grad_freq(compute_points, i, use_src=use_src)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            # local function to compute the solution at a given frequency, take only ind as argument
            fun = partial(self.compute_grad_freq, compute_points, use_src=use_src)

            res_grad = np.stack(pool.map(fun, freq_ind), axis=1)
            pool.close()

        t2 = time.time()
        if verbose:
            print("total computation time : {} s".format(t2 - t1))

        return res_grad

    def freq_to_time(self, freq_res, decay=None, low_cut=None):
        """Convert the frequency response to the time impulse response"""
        if decay is not None:
            freq_res = freq_res * np.exp(-decay * self.freqs)

        if low_cut is not None:
            b, a = signal.butter(4, low_cut, 'high', analog=True)
            _, filt = signal.freqs(b, a, worN=self.freqs)
            freq_res *= filt[np.newaxis, :]
        tval = np.linspace(0, self.nfreqs/self.fs, 2*self.nfreqs-2, endpoint=False)

        return tval, np.fft.irfft(freq_res, 2*self.nfreqs-2, axis=1)

    def compute_corner_values(self, use_src=True, freq_ind=None):

        if freq_ind is None:
            freq_ind = np.arange(self.nfreqs)

        # compute the values of the solution at the corners of the polygon (take limit for the gradient)
        res = np.zeros((len(self.poly.vertices), len(freq_ind)), dtype=complex)

        res_grad = np.zeros((len(self.poly.vertices), len(freq_ind), 2, 2), dtype=complex)
        for k, i in enumerate(freq_ind):
            mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[i]],
                                   pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                                   beta=self.mfs.beta, ampl=self.ampl[i],
                                   add_fun=self.mfs.add_fun, src_ampl=self.src_ampl[:, i])
            res[:, k] = mfs_tmp.evaluate_sol(self.poly.vertices, self.wave_numbers[i], use_src=use_src)

            res_grad[:, k, :, :] = mfs_tmp.evaluate_corner_der(self.wave_numbers[i], use_src=use_src)
        return res, res_grad

    def plot_bound_values(self, nplot):
        for i in range(1, self.nfreqs):
            mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[i]],
                                   pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                                   beta=self.mfs.beta, ampl=self.ampl[i],
                                   add_fun=self.mfs.add_fun, src_ampl=self.src_ampl[:, i])
            mfs_tmp.plot_bound_values(nplot, self.wave_numbers[i], show=False)
        plt.show()
