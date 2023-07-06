import numpy as np
from mfs import HelmholtzMFS
from utils import Polygon, c


class Simulation:
    def __init__(self, vertices, fs, nfreq, boundsep, dx, pos_src,  cond='neumann', add_src=15, nsrc_schedule=None,
                 min_dist=0.01, error_sep=None, error_thresh=0.05, sep_red=0.9):
        self.fs, self.dx = fs, dx
        self.poly = Polygon(vertices)
        self.freqs = np.linspace(0, fs, nfreq)[1:]
        # ampl will contain the amplitudes for each source at each freq, vpos the boundary positions at each mark
        self.ampl, self.vpos = [], []
        self.computed = np.zeros(nfreq, dtype=bool)

        if nsrc_schedule is None:
            # sampling strategy : 3 points per wavelength for f > 100, otherwise 10. Resample every 200 Hz or so
            fstep = self.freqs[1]
            ind100, ind200 = np.rint(100 / fstep), np.rint(200 / fstep)
            sample_marks = np.arange(ind100, nfreq-1, ind200)
            if sample_marks[-1] != nfreq - 1:
                sample_marks[-1] = nfreq - 1
            nsrc = np.zeros(len(sample_marks))
            nsrc[0], nsrc[1:] = 10, 3
            nsrc_schedule = dict(sample_marks=sample_marks, method='per_wavelength',
                                 nsrc=nsrc, max_relative_error=1e-3)
        sep = []
        freq_marks = self.freqs[nsrc_schedule["sample_marks"]]
        if nsrc_schedule["method"] == "per_wavelength":
            nsrc_schedule["sep"] = np.zeros(len(freq_marks))
        for i, f in enumerate(freq_marks):  # sample virtual boundary for each mark
            if nsrc_schedule["method"] == "per_wavelength":
                nsrc_schedule["sep"][i] = c / self.freqs[nsrc_schedule["sample_marks"][i]] / \
                                           (nsrc_schedule["sep"][i] - 1)
            boundv, normvecv, boundv_ids = self.poly.sample(nsrc_schedule["sep"][i],
                                                            sample_type="separation", min_dist=0.)
            self.vpos.append(boundv + normvecv*self.dx)

        # check if there are enough boundary points, take the max number of virtual sources if not
        if boundsep > nsrc_schedule["sep"][-1]:
            boundsep = nsrc_schedule["sep"[-1]]
            print("insufficient boundary separation to account for all frequencies, clipping")
        self.mfs = HelmholtzMFS(nbound=boundsep, nsrc=self.vpos[0], pos_src=pos_src,  # sampling boundary
                                poly=self.poly, k=self.freqs[0], src_nb="separation")

        # todo : implement parallelized solving at each remaining frequencies

        if error_sep is None:
            error_sep = boundsep / 1.1
        elif error_sep > boundsep:
            error_sep = boundsep / 1.1
            print("error_sep is larger than boundsep, clipping")

        # nodes for computing error
        error_points, _, error_ids = self.poly.sample(error_sep, sample_type="separation", min_dist=min_dist)
        for i, indfreq in enumerate(nsrc_schedule["sample_marks"]):
            self.mfs.k = 2 * np.pi * self.freqs[indfreq] / c  # update wave number
            # simulate at each mark frequency, increase number of sources until error is below threshold
            rel_error, max_tries = np.inf, 10

            if i > 0 and nsrc_schedule["sep"][i] > nsrc_schedule["sep"][i-1]:  # separation should decrease
                nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i-1]
                self.vpos[i] = self.vpos[i-1]

            while rel_error > error_thresh and max_tries > 0:
                max_tries -= 1
                self.mfs.boundv = self.vpos[i]  # update source positions

                self.mfs.mfs_fit(solver="lstsq")

                # check error
                mfs_val = self.mfs.boundary_operator(error_points, error_ids)
                src_val = -self.mfs.boundary_operator(error_points, error_ids, use_src=True)
                rel_error = np.sqrt(np.sum((np.abs(mfs_val- src_val)) ** 2)) / np.linalg.norm(src_val)

                if rel_error > error_thresh:  # resample with more sources
                    nsrc_schedule["sep"][i] = nsrc_schedule["sep"][i] * sep_red

                    self.vpos[i], _, _ = self.poly.sample(nsrc_schedule["sep"][i],
                                                          sample_type="separation", min_dist=min_dist)
                    if nsrc_schedule["sep"][i] < boundsep:
                        print("error above threshold, clipping boundary separation to minimal source separation")
                        self.mfs.bound, self.mfs.normvec, self.mfs.bound_ids = self.poly.sample(boundsep,
                                                                                                sample_type="separation",
                                                                                                min_dist=min_dist)
            if max_tries == 0:
                print("error threshold not reached, simulation might be inaccurate")
            self.ampl.append(self.mfs.ampl)