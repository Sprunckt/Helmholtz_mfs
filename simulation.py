import numpy as np
from mfs import HelmholtzMFS
from utils import Polygon, c
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation
import pickle


class Simulation:
    def __init__(self, vertices, fs, nfreq, boundsep, dx, pos_src,  cond='neumann', add_src=15, nsrc_schedule=None,
                 min_dist=0.01, error_sep=None, error_thresh=0.05, sep_red=0.8, beta=0.):
        self.fs, self.dx = fs, dx
        self.poly = Polygon(vertices)
        self.freqs = np.linspace(0, fs, nfreq)
        self.wave_numbers = 2 * np.pi * self.freqs / c
        self.nfreqs = len(self.freqs)

        # ampl will contain the amplitudes for each source at each freq, vpos the boundary positions at each mark
        self.ampl, self.vpos, self.normals = [None for _ in range(self.nfreqs)], [], []

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

        assert len(nsrc_schedule["sample_marks"]) == len(nsrc_schedule["nsrc"]) - 1, "number of sample marks should be"\
                                                                                     " one less than number of source"\
                                                                                     " parameters (last freq not in)"
        # add last sample mark (highest frequency)
        nsrc_schedule["sample_marks"] = np.concatenate([nsrc_schedule["sample_marks"], [len(self.freqs) - 1]])
        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = np.zeros(len(nsrc_schedule["sample_marks"]))
        for i in range(len(nsrc_schedule["sample_marks"])):  # sample virtual boundary for each mark
            if nsrc_schedule["method"] == "per_wavelength":
                nsrc_schedule["sep"][i] = c / self.freqs[nsrc_schedule["sample_marks"][i]] / \
                                           (nsrc_schedule["nsrc"][i] - 1)
            boundv, normvecv, _ = self.poly.sample(nsrc_schedule["sep"][i], sample_type="separation", min_dist=0.)
            self.vpos.append(boundv + normvecv*self.dx)

        # for each frequency, assign the index of the corresponding reference frequency mark
        self.freq_ref = np.zeros(nfreq, dtype=int)
        self.freq_ref[:nsrc_schedule["sample_marks"][0] + 1] = nsrc_schedule["sample_marks"][0]
        for i in range(1, len(nsrc_schedule["sample_marks"])):
            self.freq_ref[nsrc_schedule["sample_marks"][i-1] + 1:
                          nsrc_schedule["sample_marks"][i] + 1] = nsrc_schedule["sample_marks"][i]

        # convert to integers between 0 and nmarks-1
        _, self.freq_ref = np.unique(self.freq_ref, return_inverse=True)

        print(nsrc_schedule)

        # check if there are enough boundary points, take the max number of virtual sources if not
        if boundsep > nsrc_schedule["sep"][-1]:
            boundsep = nsrc_schedule["sep"][-1]
            print("insufficient boundary separation to account for all frequencies, clipping")

        # sample boundary using boundsep
        bound, normvec, bound_ids = self.poly.sample(boundsep, sample_type="separation", min_dist=min_dist)

        self.mfs = HelmholtzMFS(nbound=[bound, normvec, bound_ids], nsrc=[self.vpos[0], None], pos_src=pos_src,
                                poly=self.poly, k=self.freqs[0], cond=cond, add_fun=add_src, beta=beta,
                                sample_kwargs=dict(no_resample=True))

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
            self.mfs.k = 2 * np.pi * self.freqs[indfreq] / c  # update wave number
            # simulate at each mark frequency, increase number of sources until error is below threshold
            rel_error, max_tries = np.inf, 10

            if i > 0 and nsrc_schedule["sep"][i] > nsrc_schedule["sep"][i-1]:  # separation should decrease
                nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i-1]

                self.vpos[i] = self.vpos[i-1]
            t1 = time.time()
            while rel_error > error_thresh and max_tries > 0:
                max_tries -= 1
                self.mfs.update_sources(self.vpos[i], None)  # update source positions
                self.mfs.mfs_fit(solver="lstsq")

                # check error
                mfs_val = self.mfs.boundary_operator(error_points, error_ids)
                src_val = -self.mfs.boundary_operator(error_points, error_ids, use_src=True)
                rel_error = np.sqrt(np.sum((np.abs(mfs_val- src_val)) ** 2)) / np.linalg.norm(src_val)
                if rel_error > error_thresh:  # resample with more sources
                    nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i] * sep_red

                    self.vpos[i], normvecv, _ = self.poly.sample(nsrc_schedule["sep"][i],
                                                                 sample_type="separation", min_dist=min_dist)
                    self.vpos[i] += normvecv * self.dx
                    print("error {} above threshold, retrying "
                          "with separation {}, {} sources".format(rel_error, nsrc_schedule["sep"][i],
                                                                  len(self.vpos[i])))

                    if nsrc_schedule["sep"][i] < boundsep:
                        print("error above threshold, clipping boundary separation to minimal source separation")
                        self.mfs.update_boundary(*self.poly.sample(boundsep, sample_type="separation",
                                                                   min_dist=min_dist))
            if max_tries == 0:
                print("error threshold not reached, simulation might be inaccurate")
            self.ampl[indfreq] = self.mfs.ampl.copy()
            self.simulated[indfreq] = True
            t2 = time.time()

            print("simulating frequency {}; time {} s".format(self.freqs[indfreq], t2-t1))
            print("number of sources: {} relative error: {}".format(self.mfs.nsrc, rel_error))
        self.boundsep = boundsep

    def simulate_freq(self, ind):
        t1 = time.time()

        mfs_tmp = HelmholtzMFS(nbound=[self.mfs.bound, self.mfs.normvec, self.mfs.bound_ids],
                               nsrc=[self.vpos[self.freq_ref[ind]], None], pos_src=self.mfs.pos_src, poly=self.poly,
                               k=self.wave_numbers[ind], cond=self.mfs.cond, beta=self.mfs.beta,
                               sample_kwargs=dict(no_resample=True), add_fun=self.mfs.add_fun)
        mfs_tmp.mfs_fit(solver="lstsq")
        t2 = time.time()
        print("simulating frequency {}; time {} s; {} sources".format(self.freqs[ind], t2-t1, mfs_tmp.nsrc))
        return mfs_tmp.ampl

    def compute_freq(self, compute_points, ind):
        """Compute the solution at the given points (shape (npoints, d)) at the given frequency"""
        t1 = time.time()
        mfs_tmp = HelmholtzMFS(nbound=[self.mfs.bound, self.mfs.normvec, self.mfs.bound_ids],
                               nsrc=[self.vpos[self.freq_ref[ind]], None], pos_src=self.mfs.pos_src, poly=self.poly,
                               k=self.wave_numbers[ind], cond=self.mfs.cond, beta=self.mfs.beta, ampl=self.ampl[ind],
                               sample_kwargs=dict(no_resample=True), add_fun=self.mfs.add_fun)
        print("computing frequency {}; {} sources".format(self.freqs[ind], mfs_tmp.nsrc))
        res = mfs_tmp.evaluate_sol(compute_points)
        t2 = time.time()
        print("time {} s".format(t2 - t1))

        return res

    def simulate(self, nworkers):
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

    def compute(self, compute_points, nworkers):
        """Compute the solution at the given points (shape (npoints, d)) at all frequencies"""
        t1 = time.time()

        npoints = len(compute_points)

        if nworkers == 1:
            res = np.zeros([npoints, self.nfreqs], dtype=np.complex128)
            for i in range(1, self.nfreqs):
                res[:, i] = self.compute_freq(compute_points, i)
        else:
            if nworkers < 0:
                # use all available cores
                pool = mp.Pool()
            else:
                pool = mp.Pool(processes=nworkers)
            # local function to compute the solution at a given frequency, take only ind as argument
            fun = partial(self.compute_freq, compute_points)
            res = np.array(pool.map(fun,
                                    np.arange(1, self.nfreqs)))
            pool.close()
            # append 0 frequency
            res = np.concatenate([np.zeros([npoints, 1], dtype=np.complex128),
                                  np.reshape(res, [-1, self.nfreqs-1])], axis=1)
        t2 = time.time()
        print("total computation time : {} s".format(t2-t1))

        return res

    def freq_to_time(self, freq_res, decay=None):
        """Convert the frequency response to the time impulse response"""

        if decay is not None:
            freq_res = freq_res * np.exp(-decay * self.freqs)
        tval = np.linspace(0, self.nfreqs/self.fs, self.nfreqs, endpoint=False)

        return tval, np.fft.irfft(freq_res, self.nfreqs, axis=1)

