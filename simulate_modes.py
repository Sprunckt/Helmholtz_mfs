import numpy as np
from utils import c
from numba import jit, float64, int64, prange


@jit(float64[:, :](int64, float64[:], float64[:], float64[:, :], float64[:]), parallel=True, nopython=True)
def compute_time_sol_numba(N, dim, source_pos, compute_pos, t):
    # compute the partial sum of the eigenmode decomposition of the solution up to order N
    V, nt, d, npo = np.prod(dim), len(t), len(dim), len(compute_pos)
    xval, yval = np.arange(1, N + 1) * np.pi / dim[0], np.arange(1, N + 1) * np.pi / dim[1]
    nfreq = len(xval)

    res = np.zeros((npo, nt), dtype=np.float64)
    if d == 3:
        zval = np.arange(1, N + 1) * np.pi / dim[2]

        for ip in prange(npo):
            for it in prange(nt):
                for i in range(nfreq):
                    for j in range(nfreq):
                        for k in range(nfreq):
                            norm = np.sqrt(xval[i]**2 + yval[j]**2 + zval[k]**2)
                            res[ip, it] += 8*(c*np.sin(norm*t[it]*c)*np.cos(xval[i]*source_pos[0]) *
                                              np.cos(yval[j]*source_pos[1])*np.cos(zval[k]*source_pos[2]) *
                                              np.cos(xval[i]*compute_pos[ip, 0])*np.cos(yval[j]*compute_pos[ip, 1]) *
                                              np.cos(zval[k]*compute_pos[ip, 2]) / (norm+1e-32)) / V
                        norm = np.sqrt(xval[i] ** 2 + yval[j] ** 2)
                        res[ip, it] += 4 * (c * np.sin(norm * t[it] * c) * np.cos(xval[i] * source_pos[0]) *
                                            np.cos(yval[j] * source_pos[1]) * np.cos(xval[i] * compute_pos[ip, 0]) *
                                            np.cos(yval[j] * compute_pos[ip, 1]) / (norm + 1e-32)) / V

                        norm = np.sqrt(xval[i] ** 2 + zval[j] ** 2)
                        res[ip, it] += 4 * (c * np.sin(norm * t[it] * c) * np.cos(xval[i] * source_pos[0]) *
                                            np.cos(zval[j] * source_pos[2]) * np.cos(xval[i] * compute_pos[ip, 0]) *
                                            np.cos(zval[j] * compute_pos[ip, 2]) / (norm + 1e-32)) / V

                        norm = np.sqrt(yval[j] ** 2 + zval[i] ** 2)
                        res[ip, it] += 4 * (c * np.sin(norm * t[it] * c) * np.cos(yval[j] * source_pos[1]) *
                                            np.cos(zval[i] * source_pos[2]) * np.cos(yval[j] * compute_pos[ip, 1]) *
                                            np.cos(zval[i] * compute_pos[ip, 2]) / (norm + 1e-32)) / V
                    res[ip, it] += 2 * (c * np.sin(xval[i] * t[it] * c) * np.cos(xval[i] * source_pos[0]) *
                                        np.cos(xval[i] * compute_pos[ip, 0]) / (xval[i] + 1e-32)) / V

                    res[ip, it] += 2 * (c * np.sin(yval[i] * t[it] * c) * np.cos(yval[i] * source_pos[1]) *
                                        np.cos(yval[i] * compute_pos[ip, 1]) / (yval[i] + 1e-32)) / V

                    res[ip, it] += 2 * (c * np.sin(zval[i] * t[it] * c) * np.cos(zval[i] * source_pos[2]) *
                                        np.cos(zval[i] * compute_pos[ip, 2]) / (zval[i] + 1e-32)) / V

                res[ip, it] += c ** 2 * t[it] / V
    elif d == 2:
        for ip in prange(npo):
            for it in prange(nt):
                for i in range(nfreq):
                    for j in range(nfreq):
                        norm = np.sqrt(xval[i]**2 + yval[j]**2)
                        res[ip, it] += 4 * (c*np.sin(norm*t[it]*c) *
                                            np.cos(xval[i]*source_pos[0]) * np.cos(yval[j]*source_pos[1]) *
                                            np.cos(xval[i]*compute_pos[ip, 0]) * np.cos(yval[j]*compute_pos[ip, 1])
                                            / (norm+1e-32)) / V

                    res[ip, it] += 2 * (c * np.sin(xval[i] * t[it] * c) * np.cos(xval[i] * source_pos[0]) *
                                        np.cos(xval[i] * compute_pos[ip, 0]) / (xval[i] + 1e-32)) / V

                    res[ip, it] += 2 * (c * np.sin(yval[i] * t[it] * c) * np.cos(yval[i] * source_pos[1]) *
                                        np.cos(yval[i] * compute_pos[ip, 1]) / (yval[i] + 1e-32)) / V

                res[ip, it] += c ** 2 * t[it] / V

    return res


@jit(float64[:, :](int64, float64[:], float64[:], float64[:, :], float64[:]), parallel=True, nopython=True)
def compute_freq_sol_numba(N, dim, source_pos, compute_pos, f):
    # compute the partial sum of the eigenmode decomposition of the solution up to order N
    V, nf, d, npo = np.prod(dim), len(f), len(dim), len(compute_pos)
    res = np.zeros((npo, nf), dtype=np.float64)
    xval, yval = np.arange(1, N + 1) * np.pi / dim[0], np.arange(1, N + 1) * np.pi / dim[1]
    nfreq = len(xval)

    if d == 3:
        zval = np.arange(1, N + 1) * np.pi / dim[2]
        for ip in prange(npo):
            for it in range(nf):
                for i in range(nfreq):
                    norm1 = xval[i] ** 2
                    norm3 = zval[i] ** 2

                    for j in range(nfreq):
                        norm2 = yval[j] ** 2

                        for k in range(nfreq):
                            norm = xval[i]**2 + yval[j]**2 + zval[k]**2
                            res[ip, it] += 8*(np.cos(xval[i]*source_pos[0]) * np.cos(yval[j]*source_pos[1]) *
                                              np.cos(zval[k]*source_pos[2]) *
                                              np.cos(xval[i]*compute_pos[ip, 0]) * np.cos(yval[j]*compute_pos[ip, 1]) *
                                              np.cos(zval[k]*compute_pos[ip, 2]) / (norm - f[it]**2)) / V
                        res[ip, it] += 4 * (np.cos(xval[i] * source_pos[0]) * np.cos(xval[i] * compute_pos[ip, 0]) *
                                            np.cos(yval[j] * source_pos[1]) * np.cos(yval[j] * compute_pos[ip, 1])
                                            / (norm1 + norm2 - f[it] ** 2)) / V
                        norm3j = zval[j] ** 2
                        res[ip, it] += 4 * (np.cos(xval[i] * source_pos[0]) * np.cos(xval[i] * compute_pos[ip, 0]) *
                                            np.cos(zval[j] * source_pos[2]) * np.cos(zval[j] * compute_pos[ip, 2])
                                            / (norm1 + norm3j - f[it] ** 2)) / V
                        res[ip, it] += 4 * (np.cos(yval[j] * source_pos[1]) * np.cos(yval[j] * compute_pos[ip, 1]) *
                                            np.cos(zval[i] * source_pos[2]) * np.cos(zval[i] * compute_pos[ip, 2])
                                            / (norm2 + norm3 - f[it] ** 2)) / V
                    norm2 = yval[i] ** 2
                    res[ip, it] += 2 * (np.cos(xval[i] * source_pos[0]) * np.cos(xval[i] * compute_pos[ip, 0]) /
                                        (norm1 - f[it] ** 2)) / V
                    res[ip, it] += 2 * (np.cos(yval[i] * source_pos[1]) * np.cos(yval[i] * compute_pos[ip, 1]) /
                                        (norm2 - f[it] ** 2)) / V
                    res[ip, it] += 2 * (np.cos(zval[i] * source_pos[2]) * np.cos(zval[i] * compute_pos[ip, 2]) /
                                        (norm3 - f[it] ** 2)) / V
                res[ip, it] += 1 / (- f[it] ** 2) / V

    elif d == 2:
        for ip in prange(npo):
            for it in range(nf):
                for i in range(nfreq):
                    for j in range(nfreq):
                        norm = xval[i]**2 + yval[j]**2
                        res[ip, it] += 4*(np.cos(xval[i]*source_pos[0]) * np.cos(yval[j]*source_pos[1]) *
                                          np.cos(xval[i]*compute_pos[ip, 0]) * np.cos(yval[j]*compute_pos[ip, 1])
                                          / (norm - f[it]**2)) / V
                    norm1, norm2 = xval[i] ** 2, yval[i] ** 2
                    res[ip, it] += 2 * (np.cos(xval[i] * source_pos[0]) * np.cos(xval[i] * compute_pos[ip, 0])
                                        / (norm1 - f[it] ** 2)) / V

                    res[ip, it] += 2 * (np.cos(yval[i] * source_pos[1]) * np.cos(yval[i] * compute_pos[ip, 1])
                                        / (norm2 - f[it] ** 2)) / V

                res[ip, it] += 1 / (- f[it] ** 2) / V

    return res
