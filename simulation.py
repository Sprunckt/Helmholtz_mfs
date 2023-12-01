import numpy as np
from mfs import HelmholtzMFS, SampledBoundary
from utils import Polygon, c, ExtrudedPolygon
import multiprocessing as mp
import time
from scipy import signal
from functools import partial


class Simulation:
    def __init__(self, vertices: np.ndarray, fs: float, nfreq: int, boundsep: float, dx: float, pos_src: np.ndarray,
                 cond='neumann', add_src=15, nsrc_schedule=None, min_dist=0.01, error_sep=None, error_thresh=0.05,
                 sep_red=0.8, beta=None, height=0.):
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
        self.fs, self.dx = fs, dx
        if height > 0:
            self.poly = ExtrudedPolygon(vertices, height)
        else:
            self.poly = Polygon(vertices, origin=np.zeros(2))
        self.freqs = np.linspace(0, fs, nfreq)
        self.wave_numbers = 2 * np.pi * self.freqs / c
        self.nfreqs = len(self.freqs)
        self.min_dist = min_dist
        # ampl will contain the amplitudes for each source at each freq, vpos the boundary positions at each mark
        self.ampl, self.vpos = [None for _ in range(self.nfreqs)], []

        self.simulated = np.zeros(nfreq, dtype=bool)
        self.simulated[0] = True

        if nsrc_schedule is None:
            # sampling strategy : 3 points per wavelength for f > 100, otherwise 10. Resample every 200 Hz or so
            freq_marks = np.array([100.])
            nsrc = np.array([10, 3.])
            nsrc[0], nsrc[1:] = 10, 3
            nsrc_schedule = dict(freq_marks=freq_marks, method='per_wavelength',
                                 nsrc=nsrc, max_relative_error=1e-3)

        # generate sample_marks from freq_marks, ie indices of frequencies where to sample
        if nsrc_schedule.get("sample_marks") is None:
            nsrc_schedule["freq_marks"][-1] = np.clip(nsrc_schedule["freq_marks"][-1], 0, self.freqs[-1])

            nsrc_schedule["sample_marks"] = np.argmax(self.freqs[np.newaxis, :] >=
                                                      nsrc_schedule["freq_marks"][:, np.newaxis], axis=1)

        message = "number of sample marks should be one less than number of source parameters (last freq not in)"
        if nsrc_schedule["method"] == "per_wavelength":
            assert len(nsrc_schedule["sample_marks"]) == len(nsrc_schedule["nsrc"]) - 1, message
        elif nsrc_schedule["method"] == "separation":
            assert len(nsrc_schedule["sample_marks"]) == len(nsrc_schedule["sep"]) - 1, message
        else:
            raise ValueError("sampling method should be either per_wavelength or separation")

        # add last sample mark (highest frequency)
        nsrc_schedule["sample_marks"] = np.concatenate([nsrc_schedule["sample_marks"], [len(self.freqs) - 1]])
        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = np.zeros(len(nsrc_schedule["sample_marks"]))
        for i in range(len(nsrc_schedule["sample_marks"])):  # sample virtual boundary for each mark
            if nsrc_schedule["method"] == "per_wavelength":
                nsrc_schedule["sep"][i] = c / self.freqs[nsrc_schedule["sample_marks"][i]] / \
                                           (nsrc_schedule["nsrc"][i] - 1)
            vbound = SampledBoundary(self.poly, nsrc_schedule["sep"][i],
                                     min_dist=0., sample_type="separation", dx=self.dx)
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
        self.room_bound = SampledBoundary(self.poly, boundsep, min_dist=min_dist, sample_type="separation")

        self.mfs = HelmholtzMFS(room_bound=self.room_bound, virtual_bound=self.vpos[0], pos_src=pos_src,
                                poly=self.poly, cond=cond, add_fun=add_src, beta=beta)

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
            k = 2 * np.pi * self.freqs[indfreq] / c  # update wave number
            # simulate at each mark frequency, increase number of sources until error is below threshold
            rel_error, max_tries = np.inf, 10

            if i > 0 and nsrc_schedule["sep"][i] > nsrc_schedule["sep"][i-1]:  # separation should decrease
                nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i-1]

                self.vpos[i] = self.vpos[i-1]
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

                    self.vpos[i] = SampledBoundary(self.poly, nsrc_schedule["sep"][i],
                                                   min_dist=0., sample_type="separation", dx=self.dx)

                    print("error {} above threshold, retrying "
                          "with separation {}, {} sources".format(rel_error, nsrc_schedule["sep"][i],
                                                                  len(self.vpos[i])))

                    if nsrc_schedule["sep"][i] < boundsep:
                        boundsep = nsrc_schedule["sep"][i]*0.95
                        print("error above threshold, clipping boundary separation to minimal source separation")
                        self.mfs.update_boundary(SampledBoundary(self.poly, boundsep, min_dist=min_dist,
                                                                 sample_type="separation", dx=self.dx))
                        if error_sep > boundsep:
                            error_sep = boundsep * 0.9
                            print("error_sep is larger than boundsep, clipping")

            if max_tries == 0:
                print("error threshold not reached, simulation might be inaccurate")
            self.ampl[indfreq] = self.mfs.ampl.copy()
            self.simulated[indfreq] = True
            t2 = time.time()

            print("simulating frequency {}; time {} s".format(self.freqs[indfreq], t2-t1))
            print("number of sources: {} relative error: {}".format(self.mfs.nsrc, rel_error))
        self.boundsep = boundsep
        self.nsrc_schedule = nsrc_schedule

    def reset(self):
        self.simulated[1:] = False 

    def _resample(self):
        for i in range(len(self.vpos)):
            self.vpos[i] = SampledBoundary(self.poly, self.nsrc_schedule["sep"][i], min_dist=0., dx=self.dx)
        self.mfs.update_boundary(SampledBoundary(self.poly, self.boundsep, 
                                                 min_dist=self.min_dist, dx=self.dx))

        nsrc_max = len(self.vpos[-1])
        while nsrc_max > self.mfs.nsrc:
            self.boundsep *= 0.95
            self.mfs.update_boundary(SampledBoundary(self.poly, self.boundsep, 
                                                     min_dist=self.min_dist, dx=self.dx))

    def update_vertices(self, new_vert):
        self.poly.update_vertices(new_vert)
        self._resample()

    def update_params(self, new_phi, new_theta):
        self.poly.update_param(new_phi, new_theta)
        self._resample()

    def simulate_freq(self, ind):
        t1 = time.time()

        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly,
                               cond=self.mfs.cond, beta=self.mfs.beta, add_fun=self.mfs.add_fun)
        mfs_tmp.mfs_fit(k=self.wave_numbers[ind], solver="lstsq")
        t2 = time.time()
        print("simulating frequency {}; time {} s; {} sources".format(self.freqs[ind], t2-t1, mfs_tmp.nsrc))
        return mfs_tmp.ampl

    def compute_freq(self, compute_points, ind):
        """Compute the solution at the given points (shape (npoints, d)) at the given frequency"""
        t1 = time.time()
        mfs_tmp = HelmholtzMFS(room_bound=self.mfs.room_bound, virtual_bound=self.vpos[self.freq_ref[ind]],
                               pos_src=self.mfs.pos_src, poly=self.poly, cond=self.mfs.cond,
                               beta=self.mfs.beta, ampl=self.ampl[ind],
                               add_fun=self.mfs.add_fun)
        print("computing frequency {}; {} sources".format(self.freqs[ind], mfs_tmp.nsrc))
        res = mfs_tmp.evaluate_sol(compute_points, self.wave_numbers[ind])
        t2 = time.time()
        print("time {} s".format(t2 - t1))

        return res

    def simulate(self, nworkers=1):
        """Compute the solution coefficients at each frequency using nworkers processes."""
        t1 = time.time()
        ind_comp = np.where(~self.simulated)[0]
        if nworkers == 1:  # 1 frequency at a time
            for ind in ind_comp:
                self.ampl[ind] = self.simulate_freq(ind)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            ampl = pool.map(self.simulate_freq, ind_comp)
            pool.close()
            for i, ind in enumerate(ind_comp):
                self.ampl[ind] = ampl[i]

        print("total simulation time : {} s".format(time.time()-t1))

    def compute(self, compute_points, nworkers=1, tol=0.):
        """Compute the solution at the given points (shape (npoints, d)) at all frequencies, returns a tuple
        (res, is_in) of arrays of shape (npoints, nfreqs) and (npoints,) respectively, where is_in is True if the
        corresponding point is inside the domain, False otherwise and res contains the scalar solution at each compute
        point and frequency."""

        t1 = time.time()
        in_mask = self.poly.is_in(compute_points, min_dist=tol)

        res_tot = np.full([len(compute_points), self.nfreqs], np.nan, dtype=np.complex128)
        res_tot[in_mask, 0] = 0.  # 0 frequency

        npoints = in_mask.sum()
        ind_comp = np.where(in_mask)[0]
        to_compute = np.reshape(compute_points[ind_comp], [npoints, -1])

        if nworkers == 1:
            res = np.zeros([npoints, self.nfreqs - 1], dtype=np.complex128)
            for i in range(1, self.nfreqs):
                res[:, i] = self.compute_freq(to_compute, i)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            # local function to compute the solution at a given frequency, take only ind as argument
            fun = partial(self.compute_freq, to_compute)
            res = np.stack(pool.map(fun, np.arange(1, self.nfreqs)), axis=1)
            pool.close()

        res_tot[ind_comp, 1:] = res

        t2 = time.time()
        print("total computation time : {} s".format(t2 - t1))

        return res_tot, in_mask

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

